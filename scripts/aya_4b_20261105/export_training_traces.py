"""Export scientific loss/LR/exposure logs without server command lines or raw tokens."""
from profile_paths import resolve
import argparse, json, csv, gzip, hashlib
from pathlib import Path

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    history = []
    lr = []
    identities = {}
    orders = {}
    for role in ['fft', 'lora', 'ssu']:
        root = a.checkpoint_root / (role + ('_final_v1' if role == 'lora' else '_final_v2'))
        done = json.loads((root / 'TRAINING_DONE.json').read_text())
        assert done['steps'] == 400
        state = json.loads((root / 'trainer_state.json').read_text())
        logs = [x for x in state['log_history'] if 'loss' in x]
        assert len(logs) == 80
        for row in logs:
            history.append({'model': role, **{k: row[k] for k in ['step', 'epoch', 'loss', 'grad_norm', 'learning_rate']}})
        exposure = root / 'exposure-rank0.jsonl'
        orders[role] = [json.loads(s) for s in exposure.open()]
        assert len(orders[role]) == 6400
        identities[role] = {'actual_training_done_sha256': sha(root / 'TRAINING_DONE.json'), 'trainer_state_sha256': sha(root / 'trainer_state.json'), 'actual_exposure_sha256': sha(exposure)}
        if role != 'lora':
            candidates = list(root.glob('*applied*lr*.jsonl'))
            if not candidates:
                candidates = [root / 'optimizer-applied-lr.jsonl']
            assert len(candidates) == 1 and candidates[0].exists(), (role, [p.name for p in root.iterdir() if 'lr' in p.name])
            trace = [json.loads(s) for s in candidates[0].open()]
            assert len(trace) == 400
            for row in trace:
                lr.append({'model': role, 'optimizer_step': row['update_step'], 'actual_group_lrs': json.dumps(row['actual_lrs'], separators=(',', ':'))})
            identities[role]['actual_applied_lr_trace_sha256'] = sha(candidates[0])
    assert [[x['input_label_sha256'] for x in orders[r]] for r in ['fft', 'lora', 'ssu']].count([x['input_label_sha256'] for x in orders['fft']]) == 3
    for name, rows in [('training_curves.csv', history), ('cpuadam_applied_lr.csv', lr)]:
        with (a.output / name).open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with gzip.open(a.output / 'shared_training_exposure.jsonl.gz', 'wt') as f:
        for row in orders['fft']:
            f.write(json.dumps(row, separators=(',', ':')) + '\n')
    out = {'status': 'ACTUAL_SCIENTIFIC_TRAINING_LOGS_EXPORTED', 'training_loss_rows': len(history), 'actual_cpuadam_lr_rows': len(lr), 'shared_training_exposures': 6400, 'exposure_order_identical_for_all_three_models': True, 'source_identities': identities, 'notes': ['Loss/grad-norm/scheduler-LR logged every5steps; no interpolation', 'CPUAdam applied LR trace covers all400updates for FFT/SSU; LoRA has logged scheduler LR only', 'Shared exposure keeps hashes, step and counts only; no raw input/output token IDs', 'Independent fixed Final weights; no test-based early stop or checkpoint selection']}
    out['files_sha256'] = {p.name: sha(p) for p in a.output.iterdir() if p.is_file()}
    (a.output / 'manifest.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k in ['status', 'training_loss_rows', 'actual_cpuadam_lr_rows', 'shared_training_exposures']}))
if __name__ == '__main__':
    main()
