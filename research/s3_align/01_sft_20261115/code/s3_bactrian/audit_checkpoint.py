"""CPU audit of saved training exposure; this does not evaluate model ability."""
import argparse
import datetime
import itertools
import math

from common import *


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def audit(arm, step, destination):
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(2)
    assert not destination.exists(), 'Preserve existing audit receipts'
    lock = read(J/'EXECUTION_LOCK.json')
    assert all(sha(path) == digest for path, digest in lock.items())
    root = K/arm/'sw_final_v1'/f'checkpoint-{step}'
    files = ['accounting-rank0.json', 'exposure-rank0.jsonl',
             'auxiliary_exposure.jsonl', 'adapter_model.safetensors']
    identities = {name: sha(root/name) for name in files}
    stats = read(root/'accounting-rank0.json')
    actual = rows(root/'exposure-rank0.jsonl')
    original = K.parent/'bactrian_single_20261112/sw_final_v1/exposure-rank0.jsonl'
    with original.open() as stream:
        reference = [json.loads(line) for line in itertools.islice(stream, len(actual))]
    expected_samples = min(step*16, 65415)
    assert stats['step'] == step and stats['world_size'] == 1
    assert len(actual) == stats['trace_rows'] == expected_samples
    assert stats['trace_covers_all_local_samples']
    assert stats['exposure_file_sha256'] == identities['exposure-rank0.jsonl']
    assert stats['parameters']['trainable_unique_parameters'] == 33030144
    # These are actual inputs and assistant labels, including training order.
    assert actual == reference, 'Saved main training exposure differs from historical A'
    counts = dict(samples=len(actual),
                  input_tokens=sum(row['input_tokens'] for row in actual),
                  shifted_assistant_tokens=sum(row['shifted_assistant_tokens'] for row in actual))
    for name, value in counts.items():
        assert stats['local_counts'][name] == stats['global_counts'][name] == value

    contract_path = MATH/'keep_audit/KEEP_CONTRACT.json'
    contract = read(contract_path)
    keep_path = Path(contract['reference']['pool'])
    assert sha(keep_path) == contract['reference']['pool_sha256']
    keep_rows = rows(keep_path)
    assert [row['id'] for row in keep_rows] == contract['pool']['ordered_ids']
    align_path = J/'prepared/alignment.jsonl'
    align_rows = rows(align_path)
    meta_path = CACHE/'metadata.json'
    meta = read(meta_path)
    assert [row['id'] for row in align_rows] == meta['ordered_ids']
    assert sha(meta_path) == read(CACHE/'DONE.json')['files']['metadata.json']
    auxiliary = rows(root/'auxiliary_exposure.jsonl')
    assert len(auxiliary) == step
    for index, row in enumerate(auxiliary):
        assert row['step'] == index+1
        assert row['auxiliary_exposures_per_update'] == 1
        assert row['accelerator_GA'] == row['auxiliary_backward_divisor'] == 1
        assert math.isfinite(row['keep_loss']) and row['keep_loss'] >= 0
        if arm == 'S':
            assert row['old_id'] is None and row['old_input_tokens'] == 0
            assert row['keep_weight'] == row['keep_loss'] == 0
        else:
            old = keep_rows[index % len(keep_rows)]
            assert row['old_id'] == old['id'] and row['keep_weight'] == 1
            assert row['old_input_tokens'] == len(old['input_ids'])
        if arm == 'B':
            assert 'alignment_id' not in row
        else:
            target = align_rows[index % len(align_rows)]
            mode = 'random' if arm == 'D' else 'shared'
            assert row['alignment_id'] == target['id'] and row['alignment_mode'] == mode
            assert row['alignment_input_tokens'] == len(target['encoded']['sw']['input_ids'])
            assert math.isfinite(row['raw_alignment']) and row['raw_alignment'] >= 0
            expected = .1*row['raw_alignment']/meta['scales'][mode]
            assert math.isclose(row['alignment_loss'], expected, rel_tol=1e-5, abs_tol=1e-7)

    initial_path = K/arm/'initial-adapter-identity-v1/adapter_model.safetensors'
    initial_sha = sha(initial_path)
    assert initial_sha == 'b18a31736036aee10b7f8c5eb82e3eebd5342c6134e4b7e169c06028c34b4bec'
    weights = load_file(str(root/'adapter_model.safetensors'), device='cpu')
    initial = load_file(str(initial_path), device='cpu')
    assert weights.keys() == initial.keys()
    assert all(torch.isfinite(tensor).all() for tensor in weights.values())
    changed = sum(not torch.equal(tensor, initial[name]) for name, tensor in weights.items())
    assert changed > 200
    assert sum(tensor.numel() for tensor in weights.values()) == 33030144
    assert all(sha(root/name) == digest for name, digest in identities.items()), 'Checkpoint changed while reading'
    receipt = dict(
        status='SAVED_TRAINING_EXPOSURE_AND_ADAPTER_AUDIT_PASSED',
        at=datetime.datetime.now().astimezone().isoformat(), arm=arm, step=step,
        scientific_scope='Training integrity only; no benchmark performance or checkpoint selection.',
        main_counts=counts, main_rows_exactly_match_historical_A=True,
        auxiliary_rows=len(auxiliary), old_schedule_matches_contract=True,
        old_total_input_tokens=sum(row['old_input_tokens'] for row in auxiliary),
        alignment_total_input_tokens=sum(row.get('alignment_input_tokens', 0) for row in auxiliary),
        alignment_schedule_and_normalization_verified=arm != 'B',
        finite_adapter_tensors=len(weights), changed_adapter_tensors=changed,
        checkpoint=str(root), files=identities, historical_trace=str(original),
        historical_trace_sha256=sha(original), initial_weights_sha256=initial_sha,
        execution_lock_sha256=sha(J/'EXECUTION_LOCK.json'),
        keep_contract_sha256=sha(contract_path), alignment_pool_sha256=sha(align_path),
        teacher_metadata_sha256=sha(meta_path), audit_code_sha256=sha(__file__))
    destination.parent.mkdir(parents=True, exist_ok=True)
    dump(destination, receipt)
    print(json.dumps({key: receipt[key] for key in ['status', 'arm', 'step', 'main_counts',
                       'old_total_input_tokens', 'alignment_total_input_tokens',
                       'changed_adapter_tensors']}) + '\n' + str(destination), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--arms', nargs='+', choices=['B', 'S', 'E', 'D'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.step > 0
    for arm in args.arms:
        audit(arm, args.step, args.output/f'{arm}.json')
