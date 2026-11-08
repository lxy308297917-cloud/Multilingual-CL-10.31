"""Independent saved-artifact audit: full target stream, old pool and six nodes."""
from pathlib import Path
import os
import argparse
import collections
import hashlib
import json
import math
import torch
from safetensors import safe_open
from safetensors.torch import load_file

R = Path(__file__).resolve().parents[3]
K = Path(os.environ['P2_CHECKPOINT_ROOT'])
D = Path(os.environ['OLD_POOL_DIR'])
ORIGINAL = Path(os.environ['LEGACY_FULL_ADAPTER'])
O = R / 'research/functional_representation_20261109/p2'

def lines(p):
    return [json.loads(x) for x in p.open()]

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def main():
    torch.set_num_threads(4)
    ap = argparse.ArgumentParser();ap.add_argument('--partial', action='store_true');args = ap.parse_args()
    original = lines(ORIGINAL / 'exposure-rank0.jsonl');assert len(original) == 6400
    pool = lines(D / 'build_pool.jsonl');assert len(pool) == 128 and all(x['split'] == 'build' for x in pool)
    shape_reference = {}
    with safe_open(str(ORIGINAL / 'adapter_model.safetensors'), framework='pt') as f:
        for name in f.keys():shape_reference[name] = f.get_slice(name).get_shape()
    assert len(shape_reference) == 504
    nodes, pending, initials = [], [], {}
    for mode in ['all_layer', 'full_feature', 'ce_replay']:
        root = K / (mode + '_final_v1')
        for step in [0, 50, 100, 200, 300, 400]:
            point = root / f'checkpoint-{step}'
            ready = (point / ('INITIAL_STATE.json' if step == 0 else 'old_accounting.json')).exists()
            if not ready:
                pending.append(mode + '/' + str(step));continue
            tensors = load_file(str(point / 'adapter_model.safetensors'))
            assert tensors.keys() == shape_reference.keys()
            assert all(list(t.shape) == shape_reference[k] and torch.isfinite(t).all() for k, t in tensors.items())
            assert sum(t.numel() for t in tensors.values()) == 33030144
            nonzero_b = sum(bool(torch.count_nonzero(t)) for k, t in tensors.items() if '.lora_B.' in k)
            config = json.loads((point / 'adapter_config.json').read_text())
            assert config['r'] == 16 and config['lora_alpha'] == 32 and config['lora_dropout'] == .05
            if step == 0:
                assert nonzero_b == 0
                initials[mode] = tensors
            else:
                assert nonzero_b == 252
                target = lines(point / 'exposure-rank0.jsonl')
                expected = original[:step * 16]
                assert len(target) == len(expected) == step * 16
                assert [x['input_label_sha256'] for x in target] == [x['input_label_sha256'] for x in expected]
                for i, row in enumerate(target):
                    assert row['update_step'] == i // 16 + 1 and row['microstep'] == (i % 16) // 2 and row['row_in_microbatch'] == i % 2
                audit = json.loads((point / 'accounting-rank0.json').read_text())
                assert audit['step'] == step and audit['trace_covers_all_local_samples']
                counts = audit['global_counts'];assert counts['samples'] == step * 16
                assert counts['input_tokens'] == sum(x['input_tokens'] for x in expected)
                assert counts['shifted_assistant_tokens'] == sum(x['shifted_assistant_tokens'] for x in expected)
                padded = sum(2 * 8 * math.ceil(max(x['input_tokens'] for x in expected[i:i + 2]) / 8) for i in range(0, len(expected), 2))
                assert counts['padded_positions'] == padded
                old = lines(point / 'old_exposure.jsonl')
                assert len(old) == (0 if mode == 'all_layer' else step * 8)
                for i, x in enumerate(old):
                    source = pool[i % 128]
                    assert (x['step'], x['micro'], x['id'], x['split'], x['capability']) == (i // 8 + 1, i % 8, source['id'], 'build', source['capability'])
                    assert x['input_tokens'] == len(source['input_ids']) and x['assistant_tokens'] == len(source['input_ids']) - source['prefix_length']
                    assert x['coefficient'] == 1 and x['backward_loss_divisor'] == 8
                optimizer = torch.load(point / 'optimizer.pt', map_location='cpu', weights_only=True)
                assert len(optimizer['state']) == 504
                assert all(torch.isfinite(v).all() for value in optimizer['state'].values() for v in value.values() if isinstance(v, torch.Tensor))
                scheduler = torch.load(point / 'scheduler.pt', map_location='cpu', weights_only=True)
                assert scheduler['last_epoch'] == step
                assert (point / 'rng_state.pth').is_file() and (point / 'trainer_state.json').is_file()
            nodes.append(dict(mode=mode, step=step, adapter_sha256=sha(point / 'adapter_model.safetensors'),
                              nonzero_b_matrices=nonzero_b, parameters=33030144, tensor_shapes_and_finiteness_passed=True,
                              target_and_old_streams_checked=step > 0, optimizer_scheduler_rng_present_and_valid=step > 0))
        marker = root / 'TRAINING_DONE.json'
        if marker.exists():
            done = json.loads(marker.read_text());assert done['steps'] == 400 and done['target_order_matches_original_all6400']
            assert sha(root / 'adapter_model.safetensors') == done['weights_sha256']['adapter_model.safetensors']
            final = load_file(str(root / 'adapter_model.safetensors'));last = load_file(str(root / 'checkpoint-400/adapter_model.safetensors'))
            assert all(torch.equal(final[k], last[k]) for k in final)
    if len(initials) == 3:
        base = initials['all_layer']
        assert all(torch.equal(base[k], initials[m][k]) for m in initials for k in base)
    complete = not pending and all((K / (m + '_final_v1/TRAINING_DONE.json')).exists() for m in ['all_layer', 'full_feature', 'ce_replay'])
    if not args.partial:assert complete, pending
    report = dict(complete=complete, completed_nodes=len(nodes), expected_nodes=18, pending=pending, nodes=nodes,
                  initial_adapters_identical=len(initials) == 3, check_pool_samples_in_old_training=0,
                  source_target_exposure_sha256=sha(ORIGINAL / 'exposure-rank0.jsonl'),
                  note='0 is initial adapter only; optimizer checkpoints50/100/200/300/400; numerical recovery repeatability limitation reported separately')
    (O / 'training_artifact_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'nodes'}))

if __name__ == '__main__':
    main()
