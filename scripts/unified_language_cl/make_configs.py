"""Freeze per-track training configurations without looking at model scores."""
from runtime import runtime_path
import argparse, json, os
from pathlib import Path
import yaml
from common import ROOT, DATA, MODEL, sha, dump, rows, oh
CKPT = Path(runtime_path('checkpoints', ''))

def build(track, arm, micro=2, engineering=False):
    d = DATA / os.environ.get('UNIFIED_DATA_VERSION', 'prepared-v1') / track
    p = json.loads((d / 'protocol.json').read_text())
    steps = p['steps']
    lang = 'es' if arm == 'single_es' else 'th'
    assert steps * 16 == p['n_train_per_language']
    name = f'{track}_{arm}_seed42_v1' + (f'_engineering_micro{micro}' if engineering else '')
    src = MODEL if arm != 'seq_es_th' else CKPT / f'{track}_single_es_seed42_v1'
    cfg = dict(model_name_or_path=str(src), stage='sft', do_train=True, finetuning_type='full', template='qwen3', enable_thinking=False, train_on_prompt=False, mask_history=False, dataset_dir=str(d), dataset=lang + '_sft', tokenized_path=str(d / f'{lang}_tokenized'), cutoff_len=2048, packing=False, num_train_epochs=1.0, max_steps=steps, per_device_train_batch_size=micro, gradient_accumulation_steps=16 // (2 * micro), learning_rate=1e-05, lr_scheduler_type='cosine', warmup_ratio=0.05, optim='adamw_torch', bf16=True, gradient_checkpointing=True, seed=42, data_seed=42, save_strategy='steps', save_only_model=False, overwrite_output_dir=False, report_to='none', logging_steps=5, preprocessing_num_workers=1, dataloader_num_workers=0, accelerator_config={'use_seedable_sampler': True}, weight_decay=0.0, max_grad_norm=1.0, flash_attn='sdpa', ddp_find_unused_parameters=False, output_dir=str(CKPT / name), logging_dir=f"{runtime_path('logs', '/')}{name}/tensorboard", save_steps=100 if steps == 200 else steps // 2, deepspeed=str(ROOT / 'configs/sft_v1/deepspeed_z3.json'), full_determinism=True)
    path = Path(runtime_path('configs', '')) / '' / track / (name + '.yaml')
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    proof = dict(config_sha256=sha(path), track=track, arm=arm, engineering_only=engineering, model_revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e', initialization='same track Single ES Final, optimizer/scheduler reset' if arm == 'seq_es_th' else 'locked original official post-trained Qwen3-1.7B', protocol_sha256=sha(d / 'protocol.json'), dataset_sha256=sha(d / f'{lang}_train.jsonl'), tokenized_files_sha256={str(f.relative_to(d / f'{lang}_tokenized')): sha(f) for f in (d / f'{lang}_tokenized').rglob('*') if f.is_file()}, steps=steps, independent_examples=steps * 16, world_size=2, effective_batch=16, scheduler_reset=True, optimizer_reset=True, final_primary=True, selection_rule='fixed budget Final; no EN early stopping or test selection', common_input_semantic_order_sha256=oh([r['id'] for r in rows(d / f'{lang}_train.jsonl')]), actual_input_and_supervision=p['token_stats'][lang + ':train'], not_cross_track_compute_matched=True)
    dump(path.with_suffix('.identity.json'), proof)
    print(path, flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', required=True)
    ap.add_argument('--engineering', action='store_true')
    ap.add_argument('--micro', type=int, default=2)
    a = ap.parse_args()
    for arm in ['single_es'] if a.engineering else ['single_es', 'single_th', 'seq_es_th']:
        build(a.track, arm, a.micro, a.engineering)
