"""Freeze build-only training resources and a finite non-test-derived recipe."""
from pathlib import Path
import os
import collections
import csv
import hashlib
import json
import shutil
import numpy as np
import torch
from safetensors.torch import load_file, save_file

R = Path(__file__).resolve().parents[3]
D = Path(os.environ['P1_DIAGNOSTIC_DATA'])
O = Path(os.environ['OLD_POOL_DIR'])
C = R / 'configs/functional_representation_20261109/p2'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not O.exists()
    O.mkdir(parents=True)
    C.mkdir(parents=True, exist_ok=True)
    encoded = [json.loads(x) for x in (D / 'encoded_selected.jsonl').open()]
    build = [x for x in encoded if x['split'] == 'build']
    check_ids = {x['id'] for x in encoded if x['split'] == 'check'}
    assert len(build) == 128 and not check_ids.intersection(x['id'] for x in build)
    rng = np.random.default_rng(31415)
    groups = {cap: [x for x in build if x['capability'] == cap] for cap in ['math', 'instruction']}
    for values in groups.values():
        assert len(values) == 64
        rng.shuffle(values)
    order = [x for pair in zip(groups['math'], groups['instruction']) for x in pair]
    (O / 'build_pool.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in order))
    features = load_file(str(D / 'base_features_v1/base_features.safetensors'))['features']
    records = {x['id']: x for x in json.loads((D / 'base_features_v1/rows.json').read_text())}
    cache = {}
    for x in order:
        old = records[x['id']]
        assert old['input_ids_sha256'] == x['input_ids_sha256']
        positions = [old['feature_offset_start'] + s for s in x['answer_prediction_slots']]
        cache[x['id']] = features[positions].clone().contiguous()
    save_file(cache, str(O / 'build_reference_features.safetensors'))
    measured = list(csv.DictReader((D / 'variant_measurements_v1/full/per_example.csv').open()))
    normalizers = {cap: float(np.mean([float(x['answer_total']) for x in measured if x['split'] == 'build' and x['capability'] == cap])) for cap in groups}
    assert all(v > 0 for v in normalizers.values())
    stats = {cap: dict(examples=len(groups[cap]), input_tokens=sum(len(x['input_ids']) for x in groups[cap]),
                      assistant_tokens=sum(len(x['input_ids']) - x['prefix_length'] for x in groups[cap])) for cap in groups}
    manifest = dict(scope='P2 build-only old information; check never trained',
                    files={p.name: sha(p) for p in O.iterdir()},
                    source_encoded_sha256=sha(D / 'encoded_selected.jsonl'),
                    source_reference_sha256=sha(D / 'base_features_v1/base_features.safetensors'),
                    check_ids_used_for_training=0, build_ids=[x['id'] for x in order],
                    class_stats=stats, normalizers=normalizers,
                    normalizer_definition='Per-capability build mean full-LoRA answer_total over all36 layers; calibration-only, no formal outcome fitting',
                    feature_positions='Reference-answer first and internal prediction positions, maximum4; question slot excluded from loss')
    (O / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    plan = dict(status='ENGINEERING_PREFLIGHT', modes=['all_layer', 'full_feature', 'ce_replay'],
                target=dict(samples=6400, steps=400, effective_batch=16, seed=42, data_seed=42,
                            source_config_sha256=sha(R / 'configs/aya_4b_20261105/lora.yaml'),
                            data_lock_sha256=sha(R / 'configs/aya_4b_20261105/data.lock.json')),
                old=dict(pool=128, sampling='Fixed shuffled per-class order interleaved math/instruction, repeat cycle; one old example per target microbatch',
                         coefficient=1.0, full_feature='Mean all36 layer normalized squared drift / fixed per-capability normalizer',
                         ce_replay='Mean assistant-token CE, same old sample sequence and dropout-disabled old forwards',
                         loss_backward='Upstream target backward completed before extra old backward; old mean divided once by actual gradient accumulation steps',
                         resource_manifest_sha256=sha(O / 'manifest.json')),
                checkpoints=[0, 50, 100, 200, 300, 400], main='Final400',
                batch_choice='TO_FREEZE_FROM_ENGINEERING_THROUGHPUT_ONLY; same choice for all3',
                saving_optimizer_state=True, extra_calibration_compute_reported=True,
                functional_version='DEFERRED', formal_tests_used_for_hyperparameters=False)
    (C / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(build=128, class_stats=stats, normalizers=normalizers, check_ids_used=0)))

if __name__ == '__main__':
    main()
