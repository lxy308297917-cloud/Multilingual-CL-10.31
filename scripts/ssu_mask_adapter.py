"""Apply a locked SSU-style gradient mask before LLaMA-Factory wraps the model.

This preserves the author's gradient-hook behavior. With AdamW weight decay,
masked weights can still move; see research/ssu_port_plan.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def apply_gradient_masks(model: torch.nn.Module, mask_path: str | Path) -> dict:
    payload = json.loads(Path(mask_path).read_text())
    assert payload['format_version'] == 1
    parameters = dict(model.named_parameters())  # tied tensors appear once
    masks = payload['parameter_masks']
    assert masks and set(masks) <= set(parameters), sorted(set(masks) - set(parameters))
    frozen_elements = 0
    for name, spec in masks.items():
        parameter = parameters[name]
        assert parameter.requires_grad, name
        shape = tuple(spec['shape'])
        assert tuple(parameter.shape) == shape, (name, shape, tuple(parameter.shape))
        kind = spec['kind']
        if kind == 'columns':
            assert parameter.ndim == 2, name
            axis, width = 1, parameter.shape[1]
            multiplier = parameter.shape[0]
        elif kind == 'rows':
            assert parameter.ndim == 2, name
            axis, width = 0, parameter.shape[0]
            multiplier = parameter.shape[1]
        elif kind == 'elements':
            assert parameter.ndim == 1, name
            axis, width, multiplier = 0, parameter.numel(), 1
        else:
            raise ValueError((name, kind))
        indices = spec['indices']
        assert indices == sorted(set(indices)), name
        assert indices and 0 <= indices[0] and indices[-1] < width, name
        index_cpu = torch.tensor(indices, dtype=torch.long)

        def hook(grad: torch.Tensor, *, selected=index_cpu, dim=axis):
            return grad.index_fill_(dim, selected.to(grad.device), 0)

        parameter.register_hook(hook)
        frozen_elements += len(indices) * multiplier
    total_unique = sum(p.numel() for p in parameters.values())
    return {
        'masked_parameters': len(masks),
        'frozen_unique_elements': frozen_elements,
        'total_unique_elements': total_unique,
        'nominal_frozen_fraction': frozen_elements / total_unique,
    }
