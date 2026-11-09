def trainer_class(upstream):
    class OldCapabilityTrainer(upstream):
        old_mode = 'all_layer'
        old_rows = None
        old_references = None
        old_normalizers = None
        old_coefficient = 1.0

        def training_step(self, model, inputs, num_items_in_batch=None):
            # Finish target backward first; no simultaneous target+old activation graphs.
            target = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
            if self.old_mode == 'all_layer':
                return target
            assert self.args.world_size == 1 and self.args.n_gpu <= 1
            assert self.accelerator.gradient_accumulation_steps == 1, 'Avoid double accumulation scaling'
            step = self.state.global_step
            if getattr(self, '_old_step', None) != step:
                self._old_step, self._old_micro = step, 0
            micro = self._old_micro
            index = step * self.args.gradient_accumulation_steps + micro
            row = self.old_rows[index % len(self.old_rows)]
            assert row['split'] == 'build'
            was_training = model.training
            started = time.monotonic()
            model.eval()
            try:
                with self.compute_loss_context_manager():
                    reference = self.old_references[row['id']] if self.old_mode == 'full_feature' else None
                    normalizer = self.old_normalizers[row['capability']] if self.old_mode == 'full_feature' else 1.0
                    old = old_objective(model, row, self.old_mode, reference, normalizer)
                assert torch.isfinite(old).all(), 'Nonfinite old capability objective'
                scaled = self.old_coefficient * old / self.current_gradient_accumulation_steps
                self.accelerator.backward(scaled)
                value = float(old.detach())
            finally:
                model.train(was_training)
            if not hasattr(self, 'old_trace'):
                self.old_trace = []
            self.old_trace.append(dict(step=step + 1, micro=micro, id=row['id'], split=row['split'],
                                       capability=row['capability'], input_tokens=len(row['input_ids']),
                                       assistant_tokens=len(row['input_ids']) - row['prefix_length'],
                                       old_objective=value, coefficient=self.old_coefficient,
                                       target_loss_contribution=float(target.detach()),
                                       backward_loss_divisor=self.current_gradient_accumulation_steps,
                                       seconds_including_sync=time.monotonic() - started))
            self._old_micro += 1
            return target + scaled.detach()
    return OldCapabilityTrainer
