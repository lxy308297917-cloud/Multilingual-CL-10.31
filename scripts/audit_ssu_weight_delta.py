"""Compare selected and unselected Qwen3 weights after a masked smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def tensor_from_model(directory: Path, name: str) -> torch.Tensor:
    index_path = directory / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard = directory / index['weight_map'][name]
    else:
        shard = directory / 'model.safetensors'
    with safe_open(shard, framework='pt', device='cpu') as stream:
        return stream.get_tensor(name).float()


def summarize(values: torch.Tensor) -> dict:
    flat = values.abs().reshape(-1)
    return {
        'elements': flat.numel(),
        'changed_elements': int(torch.count_nonzero(flat).item()),
        'mean_abs_delta': float(flat.mean()),
        'max_abs_delta': float(flat.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--trained-model', type=Path, required=True)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--parameter', action='append', required=True)
    args = parser.parse_args()
    payload = json.loads(args.mask.read_text())
    result = {'mask_sha256': hashlib.sha256(args.mask.read_bytes()).hexdigest(),
              'base_model': str(args.base_model), 'trained_model': str(args.trained_model), 'parameters': {}}
    for name in args.parameter:
        spec = payload['parameter_masks'][name]
        base = tensor_from_model(args.base_model, name)
        trained = tensor_from_model(args.trained_model, name)
        if tuple(base.shape) != tuple(spec['shape']) or base.shape != trained.shape:
            raise ValueError(f'{name}: shape mismatch')
        delta = trained - base
        if spec['kind'] == 'columns':
            axis, width = 1, base.shape[1]
        elif spec['kind'] == 'rows':
            axis, width = 0, base.shape[0]
        elif spec['kind'] == 'elements':
            axis, width = 0, base.shape[0]
        else:
            raise ValueError(spec['kind'])
        selected_indices = torch.tensor(spec['indices'], dtype=torch.long)
        complement = torch.ones(width, dtype=torch.bool)
        complement[selected_indices] = False
        other_indices = torch.where(complement)[0]
        selected = delta.index_select(axis, selected_indices)
        other = delta.index_select(axis, other_indices)
        result['parameters'][name] = {
            'mask_kind': spec['kind'], 'masked': summarize(selected), 'unmasked': summarize(other),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
