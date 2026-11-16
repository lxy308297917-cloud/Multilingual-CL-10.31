"""Exact reuse of the audited historical P2 uniform36 English keep loss.

This is a loading/mode wrapper, not a replacement loss.  It delegates the
mathematical objective and the 36 block hooks to the original hash-locked
functional_representation_20261109/p2/losses.py:old_objective.

Usage with the S3 plan's target microbatch2 / GA8, single GPU:
    assets = load_keep_assets()
    loss, metadata = uniform_keep_loss(model, assets, global_step0 * 8 + micro0)
    # loss is unscaled.  After target/align backward, call exactly once:
    # accelerator.backward(assets.mu * loss / 8)
    # Require accelerator.gradient_accumulation_steps == 1 first.

One old example per target microbatch gives exactly 8 old exposures per
optimizer step.  The old sequence has 128 rows and repeats without shuffling.
No cache creation, scoring, backward, optimizer step, or GPU launch occurs here.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file


SOURCE = Path('/srv/research/10.31/multilingual-research/scripts/functional_representation_20261109/p2/losses.py')
SOURCE_SHA256 = 'd2108164df64367e787f5fe6485889099fc881c0d4d74db245c890f385961df8'
CONTRACT = Path(__file__).resolve().parent / 'keep_audit/KEEP_CONTRACT.json'


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def original_module():
    if _sha(SOURCE) != SOURCE_SHA256:
        raise RuntimeError('Historical uniform36 implementation identity changed')
    spec = importlib.util.spec_from_file_location('_s3_original_p2_uniform36', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class KeepAssets:
    rows: tuple[dict[str, Any], ...]
    references: Mapping[str, torch.Tensor]
    normalizers: Mapping[str, float]
    mu: float
    contract_sha256: str


def load_keep_assets(contract_path=CONTRACT):
    """Load the existing CPU teacher and exact old ordering after hash checks."""
    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text())
    assert contract['status'] == 'IDENTIFIED_AUDITED_P2_UNIFORM36_REUSE_CONTRACT'
    assert contract['readout']['sites'] == 36 and contract['loss']['mu'] == 1.0
    assert contract['implementation']['module_sha256'] == SOURCE_SHA256
    info = contract['reference']
    for field, digest_field in [('pool', 'pool_sha256'), ('cache', 'cache_sha256'),
                                ('manifest', 'manifest_sha256')]:
        assert _sha(info[field]) == info[digest_field], field
    manifest = json.loads(Path(info['manifest']).read_text())
    rows = tuple(json.loads(line) for line in Path(info['pool']).open())
    assert len(rows) == 128 and [r['id'] for r in rows] == contract['pool']['ordered_ids']
    assert manifest['normalizers'] == contract['loss']['normalizers']
    assert all(r['split'] == 'build' and r['question_position_slot'] not in r['answer_prediction_slots'] for r in rows)
    references = load_file(info['cache'], device='cpu')
    assert set(references) == {r['id'] for r in rows}
    for row in rows:
        tensor = references[row['id']]
        assert list(tensor.shape) == [len(row['answer_prediction_slots']), 36, 2560]
        assert not tensor.requires_grad and torch.isfinite(tensor).all()
    original_module()
    return KeepAssets(rows, references, manifest['normalizers'], 1.0, _sha(contract_path))


@contextlib.contextmanager
def _eval_with_gradients(model):
    previous = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in previous:
            module.training = training


def uniform_keep_loss(model, assets: KeepAssets, old_exposure_index: int):
    """Return original differentiable per-old-example loss, with no GA scaling.

    old_exposure_index is global_step_zero_based*8 + microbatch_index_zero_based
    for the prescribed single-GPU micro2/GA8 schedule.  The reference remains
    frozen on CPU until the original function moves it to the current device.
    """
    assert isinstance(old_exposure_index, int) and old_exposure_index >= 0
    row = assets.rows[old_exposure_index % len(assets.rows)]
    assert row['split'] == 'build'
    reference = assets.references[row['id']]
    assert not reference.requires_grad and assets.mu == 1.0
    raw = original_module().raw_model(model)
    assert len(raw.model.layers) == 36
    with _eval_with_gradients(model):
        loss = original_module().old_objective(
            model, row, 'full_feature', reference,
            assets.normalizers[row['capability']],
        )
    assert loss.requires_grad and torch.isfinite(loss)
    metadata = dict(
        version='P2-uniform36', id=row['id'], split=row['split'],
        capability=row['capability'], old_exposure_index=old_exposure_index,
        input_tokens=len(row['input_ids']),
        reference_answer_content_tokens=len(row['input_ids'])-row['prefix_length'],
        feature_positions=[row['positions'][s] for s in row['answer_prediction_slots']],
        sites=36, normalizer=assets.normalizers[row['capability']], mu=assets.mu,
        loss_already_scaled=False, reference_requires_grad=False,
    )
    return loss, metadata
