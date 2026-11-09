"""LLaMA-Factory SFT plus the narrow old-objective training-step extension."""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import json
import os
import shutil
import sys
import torch
import yaml

R = Path('/path/to/project')
D = Path('/path/to/storage/datasets/function-aware-formal-20261109/features-v1') if os.environ.get('FA_METHOD') in ['full','random','function'] else Path('/path/to/storage/datasets/functional-representation-20261109/p2-v1')
sys.path[:0] = [str(R / 'scripts/sft_v1')]
from train_formal import ExposureAccounting
from transformers import TrainerCallback
from safetensors.torch import load_file
from losses import trainer_class

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

class Audit(ExposureAccounting):
    def __init__(self, holder, expected, resume=None):
        super().__init__(resume=resume)
        self.holder, self.expected = holder, expected

    def on_train_begin(self, args, state, control, model=None, **kw):
        super().on_train_begin(args, state, control, model=model, **kw)
        assert args.world_size == 1 and args.per_device_train_batch_size * args.gradient_accumulation_steps == 16
        assert state.max_steps == self.expected
        trainer = self.holder['trainer']
        assert trainer.accelerator.gradient_accumulation_steps == 1
        if self.resume:
            from peft import get_peft_model_state_dict
            expected_weights = load_file(str(Path(self.resume) / 'adapter_model.safetensors'))
            actual_weights = get_peft_model_state_dict(model)
            assert actual_weights.keys() == expected_weights.keys()
            differences = {k: float((actual_weights[k].detach().cpu().float() - v.float()).abs().max()) for k, v in expected_weights.items()}
            assert max(differences.values()) == 0, ('Checkpoint weights changed on load', sorted(differences.items(), key=lambda x: -x[1])[:3])
            expected_optimizer = torch.load(Path(self.resume) / 'optimizer.pt', map_location='cpu', weights_only=True)
            actual_optimizer = kw['optimizer'].state_dict()
            assert actual_optimizer['param_groups'] == expected_optimizer['param_groups']
            assert actual_optimizer['state'].keys() == expected_optimizer['state'].keys()
            for key, values in expected_optimizer['state'].items():
                for name, value in values.items():
                    actual = actual_optimizer['state'][key][name]
                    assert torch.equal(actual.cpu(), value) if isinstance(value, torch.Tensor) else actual == value, (key, name, 'optimizer load changed')
            (Path(args.output_dir) / 'CHECKPOINT_LOAD_VERIFICATION.json').write_text(json.dumps({'adapter_tensors_exact_after_load': True, 'optimizer_tensors_exact_after_load': True, 'optimizer_groups_exact_after_load': True}) + '\n')
            old = Path(self.resume) / 'old_exposure.jsonl'
            assert old.exists()
            trainer.old_trace = [json.loads(x) for x in old.open()]
            for source in Path(self.resume).parent.glob('checkpoint-*'):
                if int(source.name.split('-')[-1]) <= state.global_step:
                    target = Path(args.output_dir) / source.name
                    if not target.exists():
                        target.symlink_to(source.resolve(), target_is_directory=True)
        else:
            zero = Path(args.output_dir) / 'checkpoint-0'
            zero.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(zero, safe_serialization=True)
            (zero / 'INITIAL_STATE.json').write_text(json.dumps({'optimizer_steps': 0, 'purpose': 'Initial adapter only, not a resumable optimizer state'}) + '\n')

    def on_step_end(self, args, state, control, **kw):
        super().on_step_end(args, state, control, **kw)
        control.should_save = state.global_step in ([2] if self.expected < 400 else [50, 100, 200, 300, 400])
        return control

    def record(self, args, state, path):
        super().record(args, state, path)
        trainer = self.holder['trainer']
        trace = getattr(trainer, 'old_trace', [])
        (path / 'old_exposure.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in trace))
        totals = dict(mode=trainer.old_mode, samples=len(trace),
                      input_tokens=sum(x['input_tokens'] for x in trace),
                      assistant_tokens=sum(x['assistant_tokens'] for x in trace),
                      assistant_token_count_interpretation='Reference-answer input exposure; CE labels only for ce_replay, not full_feature',
                      ce_supervised_tokens=sum(x['assistant_tokens'] for x in trace) if trainer.old_mode == 'ce_replay' else 0,
                      feature_answer_positions=sum(len(trainer.old_rows[i % len(trainer.old_rows)]['answer_prediction_slots']) for i in range(len(trace))) if trainer.old_mode == 'full_feature' else 0,
                      feature_layers=(0 if os.environ['FA_METHOD']=='kd' else 72) if trainer.old_mode == 'full_feature' else 0,
                      class_counts=dict(collections.Counter(x['capability'] for x in trace)),
                      additional_forward_backward_calls=len(trace),
                      seconds_including_sync=sum(x['seconds_including_sync'] for x in trace),
                      check_samples=0, loss_coefficient=trainer.old_coefficient,
                      selected_target_microbatch=args.per_device_train_batch_size,
                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                      loss_kwargs=trainer.model_accepts_loss_kwargs)
        (path / 'old_accounting.json').write_text(json.dumps(totals, indent=2) + '\n')

    def on_train_end(self, args, state, control, **kw):
        super().on_train_end(args, state, control, **kw)
        assert state.global_step == self.expected
        assert self.counts['samples'] == self.expected * 16

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--mode', choices=['all_layer', 'full_feature', 'ce_replay'], required=True)
    ap.add_argument('--resume', type=Path)
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    out = Path(config['output_dir'])
    assert not out.exists(), 'Independent run directories only; no overwrite'
    assert config['finetuning_type'] == 'lora' and config['gradient_checkpointing'] and not config['packing']
    assert config['learning_rate'] == 1e-4 and config['lora_rank'] == 16 and config['lora_alpha'] == 32
    assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['AYA_EXPECTED_GPU_UUID']
    assert str(torch.cuda.get_device_properties(0).uuid) == os.environ['CUDA_VISIBLE_DEVICES'].removeprefix('GPU-')
    lock = json.loads((Path(os.environ['FA_DATA_LOCK'])).read_text())
    assert all(sha(Path(config['dataset_dir']) / k) == v for k, v in lock['files'].items())
    old_manifest = json.loads((D / 'manifest.json').read_text())
    assert all(sha(D / k) == v for k, v in old_manifest['files'].items())
    rows = [json.loads(x) for x in (D / 'build_pool.jsonl').open()]
    assert len(rows) == 128 and all(x['split'] == 'build' for x in rows)
    holder = {}
    import llamafactory.train.sft.workflow as workflow
    base = trainer_class(workflow.CustomSeq2SeqTrainer)
    class ConfiguredTrainer(base):
        def __init__(self, *aa, **kk):
            super().__init__(*aa, **kk)
            self.old_mode, self.old_rows = args.mode, rows
            self.old_normalizers = old_manifest['normalizers']
            self.old_references = ({x['id']: None for x in rows} if os.environ['FA_METHOD']=='kd' else load_file(str(D / 'build_reference_features.safetensors'))) if args.mode == 'full_feature' else None
            self.old_coefficient = 1.0
            holder['trainer'] = self
    workflow.CustomSeq2SeqTrainer = ConfiguredTrainer
    if args.resume:
        assert (args.resume / 'optimizer.pt').exists() and (args.resume / 'scheduler.pt').exists()
        config['resume_from_checkpoint'] = str(args.resume)
    expected = int(config.get('max_steps', 400))
    callback = Audit(holder, expected, args.resume)
    start = datetime.datetime.now().astimezone().isoformat()
    from llamafactory.train.tuner import run_exp
    run_exp(config, callbacks=[callback])
    exposure = json.loads((out / 'accounting-rank0.json').read_text())
    old = json.loads((out / 'old_accounting.json').read_text())
    assert exposure['global_counts']['samples'] == expected * 16
    assert old['samples'] == (0 if args.mode == 'all_layer' else expected * config['gradient_accumulation_steps'])
    shutil.copy2(Path(config['model_name_or_path']) / 'generation_config.json', out / 'generation_config.json')
    result = dict(status='TRAINING_DONE' if expected == 400 else 'ENGINEERING_SMOKE_DONE', mode=os.environ['FA_METHOD'], language=os.environ['FA_LANGUAGE'],
                  steps=expected, exposure=exposure['global_counts'], old=old,
                  peak_allocated_bytes=exposure['max_memory_allocated'], elapsed_seconds=exposure['elapsed_this_process_seconds'],
                  started_at=start, ended_at=datetime.datetime.now().astimezone().isoformat(),
                  config_sha256=sha(args.config), old_manifest_sha256=sha(D / 'manifest.json'),
                  weights_sha256={p.name: sha(p) for p in out.glob('*.safetensors')},
                  recovery_source=str(args.resume) if args.resume else None,
                  saved_optimizer_state=True,
                  final_policy='Fixed Final400; no benchmark selection or English early stop')
    if expected == 400:
        assert all((out / f'checkpoint-{s}/adapter_model.safetensors').exists() for s in [0, 50, 100, 200, 300, 400])
        original = Path(os.environ['FA_SOURCE_EXPOSURE'])
        source = [json.loads(x) for x in original.open()]
        target = [json.loads(x) for x in (out / 'exposure-rank0.jsonl').open()]
        assert len(source) == len(target) == 6400
        assert [x['input_label_sha256'] for x in source] == [x['input_label_sha256'] for x in target], 'Target example/label stream changed'
        old_trace = [json.loads(x) for x in (out / 'old_exposure.jsonl').open()]
        if args.mode != 'all_layer':
            assert [x['id'] for x in old_trace] == [rows[i % 128]['id'] for i in range(expected * config['gradient_accumulation_steps'])]
        result['target_order_matches_original_all6400'] = True
        result['original_exposure_sha256'] = sha(original)
    (out / ('TRAINING_DONE.json' if expected == 400 else 'SMOKE_DONE.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
