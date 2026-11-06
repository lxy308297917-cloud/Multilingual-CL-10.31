"""Verify actual fixed-budget exposure, optimizer schedule, weights and accounting."""
from profile_paths import resolve
import argparse, json, hashlib, collections, math, csv, gzip, datetime
from pathlib import Path
import yaml
R = Path(resolve('ROOT'))
CFG = R / 'configs/aya_4b_20261105'
DATA = Path(resolve('DATA') + '/mixture-v1')
CKPT = Path(resolve('CHECKPOINTS'))
LOG = Path(resolve('LOGS'))
APP = Path(resolve('APP'))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(x)
    return h.hexdigest()

def readrows(p):
    return [json.loads(s) for s in p.open() if s.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--partial', action='store_true')
    a = p.parse_args()
    assert not a.output.exists()
    manifest = json.loads((CFG / 'data.lock.json').read_text())
    locksha = sha(CFG / 'data.lock.json')
    modelsha = sha(CFG / 'model.lock.json')
    identitydir = R / 'results/aya_4b_20261105/reproduction-inputs-v1'
    with gzip.open(identitydir / 'input_fingerprints.jsonl.gz', 'rt') as f:
        expected = [json.loads(x) for x in f]
    multiset = collections.Counter((x['input_label_sha256'] for x in expected))
    assert sum(multiset.values()) == 6400
    methods = {}
    exposures = {}
    timing = {}
    pending = []
    for role in ['fft', 'lora', 'ssu']:
        folder = CKPT / (role + ('_final_v1' if role == 'lora' else '_final_v2'))
        cfg = CFG / (role + ('.yaml' if role == 'lora' else '_v2.yaml'))
        done = folder / 'TRAINING_DONE.json'
        if not done.exists():
            pending.append(role)
            continue
        tr = json.loads(done.read_text())
        state = json.loads((folder / 'trainer_state.json').read_text())
        account = json.loads((folder / 'accounting-rank0.json').read_text())
        config = yaml.safe_load(cfg.read_text())
        assert tr['steps'] == state['global_step'] == state['max_steps'] == 400
        assert abs(state['epoch'] - 1) < 1e-08
        assert config['num_train_epochs'] == 1 and config['per_device_train_batch_size'] * config['gradient_accumulation_steps'] == 16
        assert config['seed'] == config['data_seed'] == 42
        assert config['enable_thinking'] is False and config['train_on_prompt'] is False and (config['packing'] is False)
        assert config['cutoff_len'] == 2048 and config['weight_decay'] == 0
        assert tr['config_sha256'] == sha(cfg)
        assert tr['data_manifest_sha256'] == locksha
        assert tr['model_lock_sha256'] == modelsha
        for name, digest in tr['weights_sha256'].items():
            assert sha(folder / name) == digest, name
        observed = readrows(folder / 'exposure-rank0.jsonl')
        assert len(observed) == 6400
        order = [x['input_label_sha256'] for x in observed]
        assert collections.Counter(order) == multiset
        for step in range(1, 401):
            assert sum((x['update_step'] == step for x in observed)) == 16, step
        assert all((x['rank'] == 0 for x in observed))
        counts = account['global_counts']
        assert counts['samples'] == 6400 and counts['input_tokens'] == manifest['input_tokens'] == 2373891 and (counts['shifted_assistant_tokens'] == manifest['shifted_assistant_tokens'] == 324892)
        assert sum((x['input_tokens'] for x in observed)) == counts['input_tokens']
        assert sum((x['shifted_assistant_tokens'] for x in observed)) == counts['shifted_assistant_tokens']
        exposures[role] = order

        def wanted(step):
            k = step - 1
            return config['learning_rate'] * (k / 20 if k < 20 else 0.5 * (1 + math.cos(math.pi * (k - 20) / 380)))
        history = [x for x in state['log_history'] if 'loss' in x]
        assert len(history) == 80
        for h in history:
            assert math.isfinite(h['loss']) and math.isfinite(h['grad_norm']) and math.isclose(h['learning_rate'], wanted(h['step']), rel_tol=1e-09, abs_tol=1e-14)
        applied = folder / 'applied-lr-rank0.jsonl'
        if role != 'lora':
            trace = readrows(applied)
            assert [x['update_step'] for x in trace] == list(range(1, 401))
            for x in trace:
                assert all((math.isclose(v, wanted(x['update_step']), rel_tol=1e-09, abs_tol=1e-14) for v in x['actual_lrs']))
            binding = json.loads((folder / 'optimizer-binding.json').read_text())
            assert binding['actual_optimizer_type'] == 'DeepSpeedCPUAdam' and binding['shared_group_dictionaries_after']
            assert all((v == 0 for v in binding['actual_weight_decays']))
        methods[role] = {'global_step': 400, 'epochs': 1, 'samples': 6400, 'input_tokens': counts['input_tokens'], 'shifted_assistant_tokens': counts['shifted_assistant_tokens'], 'allocated_sequence_positions': counts['padded_positions'], 'padding_positions': counts['padded_positions'] - counts['input_tokens'], 'raw_counter_note': 'Historical padded_positions counter means all allocated sequence positions; padding is the derived difference', 'parameters': account['parameters'], 'max_gpu_allocated_bytes': account['max_memory_allocated'], 'max_gpu_reserved_bytes': account['max_memory_reserved'], 'actual_saved_weights_sha256': tr['weights_sha256'], 'config_sha256': sha(cfg), 'training_done_sha256': sha(done), 'exposure_sha256': sha(folder / 'exposure-rank0.jsonl'), 'input_label_multiset_matches_frozen_data': True, 'steps_each_exposed16_examples': True, 'logged_lr_and_finite_loss_checked': 80, 'applied_cpuadam_lr_steps_checked': 400 if role != 'lora' else None, 'saved_optimizer_state': False}
        metrics = json.loads((folder / 'train_results.json').read_text())
        timing[role] = {'trainer_runtime_seconds': metrics['train_runtime'], 'process_wall_seconds': (datetime.datetime.fromisoformat(tr['ended_at']) - datetime.datetime.fromisoformat(tr['started_at'])).total_seconds(), 'samples_per_second': metrics['train_samples_per_second'], 'optimizer_steps_per_second': metrics['train_steps_per_second'], 'started_at': tr['started_at'], 'ended_at': tr['ended_at']}
    if not a.partial:
        assert set(methods) == {'fft', 'lora', 'ssu'} and exposures['fft'] == exposures['lora'] == exposures['ssu']
    order_equal = all((value == next(iter(exposures.values())) for value in exposures.values())) if exposures else None
    a.output.mkdir(parents=True)
    result = {'status': 'VERIFIED_COMPLETE' if not a.partial else 'PARTIAL_NOT_FINAL', 'methods': methods, 'pending': pending, 'full_example_order_equal_among_completed_methods': order_equal, 'input_tokens': manifest['input_tokens'], 'shifted_assistant_tokens': manifest['shifted_assistant_tokens'], 'model_lock_sha256': modelsha, 'data_manifest_sha256': locksha, 'baseline_selection': 'All methods fixed Final400, no test-based checkpoint selection'}
    (a.output / 'training_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (a.output / 'training_cost.json').write_text(json.dumps(timing, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'completed_methods': list(methods), 'pending': pending, 'exposure_order_equal': order_equal}))
if __name__ == '__main__':
    main()
