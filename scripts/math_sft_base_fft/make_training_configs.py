"""Freeze one independent Base initialization per selected language."""
from profile_paths import resolve
import argparse, math, shutil, json, os
from pathlib import Path
from common import *
import yaml
ap = argparse.ArgumentParser()
ap.add_argument('--langs', nargs='+', required=True)
a = ap.parse_args()
assert len(set(a.langs)) == len(a.langs)
checkpoint = Path(resolve('CHECKPOINTS'))
checkpoint.mkdir(exist_ok=True)
link = Path(resolve('CHECKPOINTS'))
if link.exists():
    assert link.resolve() == checkpoint
else:
    link.symlink_to(checkpoint, target_is_directory=True)
assert shutil.disk_usage(checkpoint).free > 35 * 1024 ** 3, 'Need full A route storage; do not cut training budget.'
for lang in a.langs:
    data = DATA / 'training-v1' / lang
    info = json.loads((data / 'training_data.json').read_text())
    name = f'mathoctopus_{lang}_full_2epochs_seed42_v1'
    out = checkpoint / name
    cfg = {'model_name_or_path': str(MODEL), 'stage': 'sft', 'do_train': True, 'finetuning_type': 'full', 'template': 'qwen3', 'enable_thinking': False, 'train_on_prompt': False, 'mask_history': False, 'dataset_dir': str(data), 'dataset': lang + '_sft', 'tokenized_path': str(data / 'tokenized'), 'cutoff_len': 2048, 'packing': False, 'num_train_epochs': 2.0, 'max_steps': -1, 'per_device_train_batch_size': 2, 'gradient_accumulation_steps': 4, 'learning_rate': 5e-06, 'lr_scheduler_type': 'cosine', 'warmup_ratio': 0.05, 'optim': 'adamw_torch', 'bf16': True, 'gradient_checkpointing': True, 'seed': 42, 'data_seed': 42, 'save_strategy': 'epoch', 'save_only_model': True, 'overwrite_output_dir': False, 'report_to': 'none', 'logging_steps': 5, 'preprocessing_num_workers': 1, 'dataloader_num_workers': 0, 'dataloader_drop_last': False, 'accelerator_config': {'use_seedable_sampler': True}, 'weight_decay': 0.0, 'max_grad_norm': 1.0, 'flash_attn': 'sdpa', 'ddp_find_unused_parameters': False, 'output_dir': str(out), 'logging_dir': str(Path(resolve('LOGS')) / name / 'tensorboard'), 'deepspeed': str(ROOT / 'configs/sft_v1/deepspeed_z3.json'), 'full_determinism': True}
    p = ROOT / 'configs/math_sft_base_fft' / (name + '.yaml')
    assert not p.exists()
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    dump(p.with_suffix('.budget.json'), {'language': lang, 'samples_per_unpadded_epoch': info['samples'], 'expected_steps_per_epoch': math.ceil(info['samples'] / 16), 'expected_total_steps': 2 * math.ceil(info['samples'] / 16), 'epoch1_step': math.ceil(info['samples'] / 16), 'epoch2_step': 2 * math.ceil(info['samples'] / 16), 'data_manifest_sha256': sha(data / 'training_data.json'), 'config_sha256': sha(p), 'optimizer_state_persisted': False, 'continuous_optimizer_scheduler_in_memory': True, 'budget': 'all valid records independently, two full passes; final partial effective batch and any DDP padding reported honestly', 'epoch_checkpoints_have_full_weights': True, 'gpu_count': 2, 'effective_batch': 16})
    print(p, info['samples'], 2 * math.ceil(info['samples'] / 16), flush=True)
