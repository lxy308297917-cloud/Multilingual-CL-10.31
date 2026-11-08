"""CPU oracle: upstream target gradients plus separately accumulated old gradients."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import copy
import json
from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments, Qwen3Config, Qwen3ForCausalLM
from peft import get_peft_model, LoraConfig
from losses import trainer_class, old_objective, raw_model


def cached(model, row):
    raw = raw_model(model)
    values, handles = [], []
    positions = [row['positions'][j] for j in row['answer_prediction_slots']]
    for block in raw.model.layers:
        handles.append(block.register_forward_hook(lambda m, a, b: values.append(b[0, positions].detach())))
    with torch.no_grad():
        raw.model(input_ids=torch.tensor([row['input_ids']]), use_cache=False)
    for h in handles:
        h.remove()
    return torch.stack(values, dim=1)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                      attention_dropout=0)
    cfg._attn_implementation = 'eager'
    model = get_peft_model(Qwen3ForCausalLM(cfg), LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')).eval()
    rows = [dict(id=str(i), split='build', capability='math' if i == 0 else 'instruction',
                 input_ids=[3, 5, 7, 9, 11, 13, 15][:6 + i], prefix_length=3,
                 positions=[1, 2, 4], answer_prediction_slots=[1, 2]) for i in range(2)]
    references = {x['id']: cached(model, x) for x in rows}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'lora_B' in name:
                p.normal_(std=.03)
    results = []
    for mode in ['all_layer', 'ce_replay', 'full_feature']:
        for accepts in [False, True]:
            for checkpointing in [False, True]:
                actual, oracle = copy.deepcopy(model), copy.deepcopy(model)
                if checkpointing:
                    actual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                    oracle.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                args = TrainingArguments(output_dir='/tmp/functional-representation-cpu-oracle', use_cpu=True,
                                         per_device_train_batch_size=1, gradient_accumulation_steps=2,
                                         report_to='none', disable_tqdm=True)
                cls = trainer_class(Trainer)
                trainer = cls(model=actual, args=args)
                trainer.model_accepts_loss_kwargs = accepts
                trainer.current_gradient_accumulation_steps = 2
                trainer.old_mode, trainer.old_rows, trainer.old_references = mode, rows, references
                trainer.old_normalizers = {'math': .05, 'instruction': .05}
                targets = []
                for ids in [[2, 4, 6, 8, 10, 12], [3, 6, 9, 12, 15, 18, 21, 24]]:
                    t = torch.tensor([ids]);labels = t.clone();labels[:, :3] = -100
                    targets.append(dict(input_ids=t, labels=labels, attention_mask=torch.ones_like(t)))
                total = sum(int((x['labels'][:, 1:] != -100).sum()) for x in targets)
                for batch in targets:
                    trainer.training_step(actual, batch, num_items_in_batch=torch.tensor(total))
                for batch, row in zip(targets, rows):
                    oracle.train()
                    loss = oracle(**batch, **({'num_items_in_batch': torch.tensor(total)} if accepts else {})).loss
                    (loss if accepts else loss / 2).backward()
                    if mode != 'all_layer':
                        oracle.eval()
                        old = old_objective(oracle, row, mode, references[row['id']] if mode == 'full_feature' else None, .05)
                        (old / 2).backward()
                maximum = 0.
                for (na, a), (nb, b) in zip(actual.named_parameters(), oracle.named_parameters()):
                    assert na == nb
                    if a.requires_grad:
                        assert a.grad is not None and b.grad is not None
                        error = float((a.grad - b.grad).abs().max());maximum = max(maximum, error)
                        assert torch.allclose(a.grad, b.grad, atol=2e-7, rtol=2e-5), (mode, accepts, checkpointing, na, error)
                    else:
                        assert a.grad is None
                assert actual.training
                assert all(not block._forward_hooks for block in raw_model(actual).model.layers)
                assert len(getattr(trainer, 'old_trace', [])) == (0 if mode == 'all_layer' else 2)
                results.append(dict(mode=mode, model_accepts_loss_kwargs=accepts,
                                    gradient_checkpointing=checkpointing, maximum_gradient_error=maximum))
    Path('accumulation_validation.json').write_text(json.dumps(
        dict(scope='CPU tiny Qwen3+PEFT engineering oracle; not efficacy', cases=results,
             accumulation_steps=2, variable_target_supervised_lengths=True,
             upstream_target_backward_unchanged=True, extra_old_backward_scaled_once=True), indent=2) + '\n')
    print('Gradient accumulation, old forward mode, checkpoint recomputation and hook cleanup verified:', len(results), 'cases')

if __name__ == '__main__':
    main()
