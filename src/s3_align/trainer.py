"""Thin LLaMA-Factory extension; one auxiliary exposure per optimizer step."""
import json
import math
from pathlib import Path

import torch
from .losses import alignment, eval_grad, old_objective


def make_trainer(upstream, assets, arm):
    if arm not in ('T0', 'T1', 'T2'):
        raise ValueError(arm)

    class JointTrainer(upstream):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_aux = -1
            self.aux = []

        def training_step(self, model, inputs, num_items_in_batch=None):
            target = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
            assert self.accelerator.gradient_accumulation_steps == 1
            assert self.args.world_size == 1
            step = int(self.state.global_step)
            if step == self.last_aux:
                return target
            self.last_aux = step
            row = assets.rows[step % len(assets.rows)]
            with eval_grad(model), self.compute_loss_context_manager():
                keep = old_objective(model, row, 'full_feature', assets.references[row['id']],
                                     assets.normalizers[row['capability']])
            assert keep.requires_grad and torch.isfinite(keep)
            self.accelerator.backward(keep)
            rec = dict(step=step + 1, keep_id=row['id'], keep_loss=float(keep.detach()),
                       keep_input_tokens=len(row['input_ids']),
                       keep_scale=assets.normalizers[row['capability']], mu=1.0)
            total = target + keep.detach()
            del keep
            if arm != 'T0':
                index = step % len(assets.alignment_rows)
                paired = assets.alignment_rows[index]
                with eval_grad(model), self.compute_loss_context_manager():
                    raw = alignment(model, paired['encoded']['sw'], index, arm, assets.maps)
                    loss = assets.coefficient * raw
                    assert loss.requires_grad and torch.isfinite(loss)
                    if step in [0, 100, 500, 2000, 4000]:
                        params = [p for p in model.parameters() if p.requires_grad]
                        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                        rec['alignment_gradient_norm'] = math.sqrt(sum(
                            float(g.double().square().sum()) for g in grads if g is not None))
                    self.accelerator.backward(loss)
                rec.update(alignment_id=paired['id'], alignment_raw=float(raw.detach()),
                           alignment_weighted=float(loss.detach()), lambda_fixed=assets.coefficient,
                           alignment_input_tokens=len(paired['encoded']['sw']['input_ids']))
                total += loss.detach()
            self.aux.append(rec)
            out = Path(self.args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with (out / 'auxiliary_live.jsonl').open('a') as stream:
                stream.write(json.dumps(rec) + '\n')
            return total
    return JointTrainer
