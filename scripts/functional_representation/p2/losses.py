"""Thin extra-old-batch training step; target SFT stays in the upstream trainer."""
import time
import torch
import torch.nn.functional as F


def raw_model(model):
    return model.get_base_model() if hasattr(model, 'get_base_model') else model


def old_objective(model, row, mode, reference=None, normalizer=1.0):
    """Old forward uses eval mode but retains adapter gradients; caller restores mode."""
    raw = raw_model(model)
    ids = torch.tensor([row['input_ids']], dtype=torch.long, device=next(raw.parameters()).device)
    if mode == 'ce_replay':
        hidden = raw.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        p = row['prefix_length']
        logits = raw.lm_head(hidden[:, p - 1:-1, :]).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, p:].reshape(-1))
    assert mode == 'full_feature' and reference is not None and normalizer > 0
    states, hooks = [], []
    slots = row['answer_prediction_slots']
    positions = [row['positions'][slot] for slot in slots]
    for block in raw.model.layers:
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            states.append(h[0, positions, :])
        hooks.append(block.register_forward_hook(hook))
    try:
        raw.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in hooks:
            handle.remove()
    assert len(states) == len(raw.model.layers)
    current = torch.stack(states, dim=1).float()  # answer positions x layers x features
    ref = reference.to(device=current.device, dtype=torch.float32)
    assert current.shape == ref.shape
    per_layer = (current - ref).square().sum(dim=(0, 2)) / ref.square().sum(dim=(0, 2)).clamp_min(1e-12)
    return per_layer.mean() / normalizer


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
