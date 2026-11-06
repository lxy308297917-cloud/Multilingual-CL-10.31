"""Check completed revised LoRA actual exposures against its own frozen capsule."""
from profile_paths import resolve
import argparse, json, hashlib, gzip, collections, math, datetime
from pathlib import Path
import yaml

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--identities', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    root = a.checkpoint
    done = json.loads((root / 'TRAINING_DONE.json').read_text())
    state = json.loads((root / 'trainer_state.json').read_text())
    cfg = yaml.safe_load(a.config.read_text())
    manifest = Path(cfg['dataset_dir']) / 'manifest.json'
    data = json.loads(manifest.read_text())
    spec = json.loads((a.identities / 'reproduction_spec.json').read_text())
    assert done['steps'] == state['global_step'] == state['max_steps'] == 400 and done['samples'] == 6400 and (abs(state['epoch'] - 1) < 1e-08)
    assert cfg['seed'] == cfg['data_seed'] == 42 and cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps'] == 16
    assert done['config_sha256'] == sha(a.config) and done['data_manifest_sha256'] == sha(manifest)
    assert cfg['enable_thinking'] is False and cfg['train_on_prompt'] is False and (cfg['packing'] is False) and (cfg['cutoff_len'] == 2048)
    expected = [json.loads(x) for x in gzip.open(a.identities / 'input_fingerprints.jsonl.gz', 'rt')]
    actual = [json.loads(x) for x in (root / 'exposure-rank0.jsonl').open()]
    assert len(actual) == len(expected) == 6400
    assert collections.Counter((x['input_label_sha256'] for x in actual)) == collections.Counter((x['input_label_sha256'] for x in expected))
    steps = collections.Counter((x['update_step'] for x in actual))
    assert steps == {i: 16 for i in range(1, 401)}
    assert all((x['rank'] == 0 for x in actual))
    counts = json.loads((root / 'accounting-rank0.json').read_text())['global_counts']
    assert counts['input_tokens'] == data['input_tokens'] == spec['input_tokens'] == sum((x['input_tokens'] for x in actual))
    assert counts['shifted_assistant_tokens'] == data['shifted_assistant_tokens'] == spec['shifted_assistant_tokens'] == sum((x['shifted_assistant_tokens'] for x in actual))
    history = [x for x in state['log_history'] if 'loss' in x]
    assert len(history) == 80
    for x in history:
        step = x['step'] - 1
        expected_lr = cfg['learning_rate'] * (step / 20 if step < 20 else 0.5 * (1 + math.cos(math.pi * (step - 20) / 380)))
        assert math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) and math.isclose(x['learning_rate'], expected_lr, rel_tol=1e-09, abs_tol=1e-14)
    for filename, digest in done['weights_sha256'].items():
        assert sha(root / filename) == digest
    metrics = json.loads((root / 'train_results.json').read_text())
    out = {'status': 'VERIFIED_REVISED_LORA_FINAL400', 'samples': 6400, 'steps': 400, 'input_tokens': counts['input_tokens'], 'shifted_assistant_tokens': counts['shifted_assistant_tokens'], 'all_input_label_exposures_match_frozen_candidate': True, 'samples_per_optimizer_step': 16, 'finite_loss_gradient_and_scheduler_logs': 80, 'learning_rate': cfg['learning_rate'], 'data_manifest_sha256': sha(manifest), 'config_sha256': sha(a.config), 'weights_sha256': done['weights_sha256'], 'actual_exposure_sha256': sha(root / 'exposure-rank0.jsonl'), 'trainer_runtime_seconds': metrics['train_runtime'], 'process_wall_seconds': (datetime.datetime.fromisoformat(done['ended_at']) - datetime.datetime.fromisoformat(done['started_at'])).total_seconds(), 'optimizer_state_saved': False}
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))
if __name__ == '__main__':
    main()
