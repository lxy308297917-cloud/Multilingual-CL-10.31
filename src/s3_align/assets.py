"""Load explicitly supplied, hash-checked frozen experiment assets."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math

import torch
from safetensors.torch import load_file


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    # Physical LF only: U+2028 inside a JSON string must not split a record.
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Assets:
    rows: list
    references: dict
    normalizers: dict
    alignment_rows: list
    maps: dict
    coefficient: float


def load_assets(spec_path, arm):
    if arm not in ('T0', 'T1', 'T2'):
        raise ValueError(arm)
    spec_path = Path(spec_path).resolve()
    spec = read_json(spec_path)
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else spec_path.parent / path
    # Rebinding a server path requires a new deployment manifest, not a new
    # scientific dataset. The original archive contract remains unchanged.
    def checked(key):
        path = resolve(spec[key]['path'])
        if sha256(path) != spec[key]['sha256']:
            raise ValueError(f'Asset hash mismatch: {key}')
        return path
    contract = read_json(checked('keep_contract'))
    if contract['status'] != 'IDENTIFIED_AUDITED_P2_UNIFORM36_REUSE_CONTRACT':
        raise ValueError('Unsupported English preservation contract')
    if contract['readout']['sites'] != 36 or contract['loss']['mu'] != 1.0:
        raise ValueError('This recipe preserves the historical uniform36 objective')
    rows = read_jsonl(checked('keep_pool'))
    refs = load_file(str(checked('keep_cache')), device='cpu')
    reference = contract['reference']
    manifest = read_json(checked('keep_manifest'))
    assert spec['keep_manifest']['sha256'] == reference['manifest_sha256']
    assert manifest['normalizers'] == contract['loss']['normalizers']
    assert all(math.isfinite(v) and v > 0 for v in manifest['normalizers'].values())
    assert spec['keep_pool']['sha256'] == reference['pool_sha256']
    assert spec['keep_cache']['sha256'] == reference['cache_sha256']
    assert len(rows) == 128
    assert [r['id'] for r in rows] == contract['pool']['ordered_ids']
    assert set(refs) == {r['id'] for r in rows}
    for row in rows:
        assert row['split'] == 'build'
        assert row['question_position_slot'] not in row['answer_prediction_slots']
        tensor = refs[row['id']]
        assert list(tensor.shape) == [len(row['answer_prediction_slots']), 36, 2560]
        assert not tensor.requires_grad and torch.isfinite(tensor).all()
    if arm == 'T0':
        return Assets(rows, refs, contract['loss']['normalizers'], [], {}, 0.0)
    alignment_rows = read_jsonl(checked('alignment_pool'))
    maps = load_file(str(checked('mappings')), device='cpu')
    calibration = read_json(checked('gradient_calibration'))
    assert len(alignment_rows) == 1024
    coefficient = calibration['lambda'][arm]
    assert 0 < coefficient <= 10
    for key in ['body.mean.sw', f'{arm}.projection.sw', f'{arm}.std.sw', f'{arm}.teacher']:
        assert torch.isfinite(maps[key]).all()
    assert list(maps['body.mean.sw'].shape) == [2560]
    assert list(maps[f'{arm}.projection.sw'].shape) == [2560, 64]
    assert list(maps[f'{arm}.std.sw'].shape) == [64]
    assert (maps[f'{arm}.std.sw'] > 0).all()
    assert list(maps[f'{arm}.teacher'].shape) == [1024, 64]
    return Assets(rows, refs, contract['loss']['normalizers'], alignment_rows, maps, coefficient)
