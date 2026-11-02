"""Build an auditable Qwen3 SSU-Wanda mask from locked English calibration blocks.

Matches the author's column score (L2 weight column norm times mean squared
input activation) and 50% top-column selection. Unlike the author's fallback,
missing or incompatible activations fail the run. This script never trains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def selected_indices(weight: torch.Tensor, importance: torch.Tensor, ratio: float) -> list[int]:
    """Top columns by the author's |W| column norm * E[x^2] score."""
    if weight.ndim != 2 or importance.numel() != weight.shape[1]:
        raise ValueError((tuple(weight.shape), tuple(importance.shape)))
    count = int(weight.shape[1] * ratio)
    scores = torch.linalg.vector_norm(weight.detach().float().abs(), dim=0) * importance.float().to(weight.device)
    if not torch.isfinite(scores).all():
        raise ValueError('nonfinite Wanda scores')
    return sorted(torch.topk(scores.cpu(), count, largest=True).indices.tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--num-blocks', type=int, default=500)
    parser.add_argument('--block-length', type=int, default=2048)
    parser.add_argument('--freeze-ratio', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists() or not (0 < args.freeze_ratio < 1) or args.num_blocks < 1:
        raise ValueError('output exists, invalid ratio, or invalid block count')
    dataset = load_from_disk(str(args.calibration))
    if len(dataset) < args.num_blocks:
        raise ValueError(f'{len(dataset)} blocks < requested {args.num_blocks}')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dtype = torch.bfloat16 if args.device.startswith('cuda') else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation='sdpa',
        device_map={'': args.device}, local_files_only=True,
    )
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device
    accumulators: dict[str, torch.Tensor] = {}
    calls: dict[str, int] = defaultdict(int)
    handles = []

    def hook_for(name: str):
        def hook(_module, inputs, _output):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ValueError(f'no activation for {name}')
            values = inputs[0].detach().float()
            if not values.is_floating_point() or values.ndim < 2:
                raise ValueError(f'invalid activation for {name}: {tuple(values.shape)}')
            mean_square = values.square().mean(dim=tuple(range(values.ndim - 1)))
            if name not in accumulators:
                accumulators[name] = mean_square
            else:
                accumulators[name] += mean_square
            calls[name] += 1
        return hook

    parameters = dict(model.named_parameters())
    monitored = []
    for name, module in model.named_modules():
        own = list(module.named_parameters(recurse=False))
        if not own or name.endswith('embed_tokens') or name == 'lm_head':
            continue
        if any(parameter.ndim not in (1, 2) for _, parameter in own):
            raise ValueError(f'unexpected parameter rank in {name}')
        handles.append(module.register_forward_hook(hook_for(name)))
        monitored.append(name)

    try:
        with torch.inference_mode():
            for index in range(args.num_blocks):
                row = dataset[index]
                input_ids = torch.tensor(row['input_ids'], dtype=torch.long, device=device)[None, :]
                attention_mask = torch.tensor(row['attention_mask'], dtype=torch.long, device=device)[None, :]
                if input_ids.shape[1] != args.block_length or not torch.all(attention_mask == 1):
                    raise ValueError(f'calibration block {index} is not complete')
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                if (index + 1) % 50 == 0:
                    print(f'calibration_blocks={index + 1}/{args.num_blocks}', flush=True)
    finally:
        for handle in handles:
            handle.remove()

    if set(accumulators) != set(monitored) or any(calls[name] != args.num_blocks for name in monitored):
        raise ValueError('missing or repeated activation module call')

    masks: dict[str, dict] = {}
    diagnostics = {}
    for name, module in model.named_modules():
        if name not in accumulators:
            continue
        importance = accumulators[name].div(args.num_blocks).cpu()
        if not torch.isfinite(importance).all():
            raise ValueError(f'nonfinite activation importance: {name}')
        for local_name, parameter in module.named_parameters(recurse=False):
            full_name = f'{name}.{local_name}'
            if full_name not in parameters:
                raise ValueError(f'parameter name mismatch: {full_name}')
            if parameter.ndim == 2:
                indices = selected_indices(parameter, importance, args.freeze_ratio)
                kind = 'columns'
            else:
                if importance.numel() != parameter.numel():
                    raise ValueError(f'activation/weight size mismatch: {full_name}')
                scores = parameter.detach().float().abs().cpu() * (1 + importance)
                indices = sorted(torch.topk(scores, int(parameter.numel() * args.freeze_ratio)).indices.tolist())
                kind = 'elements'
            if indices:
                masks[full_name] = {'kind': kind, 'shape': list(parameter.shape), 'indices': indices}
            diagnostics[full_name] = {
                'activation_mean_square_min': float(importance.min()),
                'activation_mean_square_max': float(importance.max()),
                'selected_count': len(indices),
            }

    special_ids = sorted(set(tokenizer.all_special_ids) | {
        token_id for token in ('<|im_start|>', '<|im_end|>', '<think>', '</think>')
        if (token_id := tokenizer.convert_tokens_to_ids(token)) is not None
    })
    embedding_name = 'model.embed_tokens.weight'
    embedding = parameters[embedding_name]
    if not all(0 <= token_id < embedding.shape[0] for token_id in special_ids):
        raise ValueError('special token ID outside tied embedding')
    masks[embedding_name] = {'kind': 'rows', 'shape': list(embedding.shape), 'indices': special_ids}
    masked_count = sum(
        len(spec['indices']) * (spec['shape'][0] if spec['kind'] == 'columns' else
                                spec['shape'][1] if spec['kind'] == 'rows' else 1)
        for spec in masks.values()
    )
    payload = {
        'format_version': 1,
        'algorithm': 'author_ssu_wanda_qwen3_source_calibration',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_config_sha256': file_sha256(args.model / 'config.json'),
        'tokenizer_json_sha256': file_sha256(args.model / 'tokenizer.json'),
        'calibration_state_sha256': file_sha256(args.calibration / 'state.json'),
        'num_blocks': args.num_blocks,
        'block_length': args.block_length,
        'freeze_ratio_per_matrix': args.freeze_ratio,
        'statistic': f'mean(x^2) per complete {args.block_length}-token block; then arithmetic mean across blocks',
        'score': 'L2(abs(weight), output_axis) * input_mean_square',
        'special_token_ids': special_ids,
        'parameter_masks': dict(sorted(masks.items())),
        'diagnostics': dict(sorted(diagnostics.items())),
        'frozen_unique_elements_nominal': masked_count,
        'total_unique_elements': sum(parameter.numel() for parameter in parameters.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
        stream.write('\n')
    print(json.dumps({
        'out': str(args.out), 'sha256': file_sha256(args.out),
        'masked_parameters': len(masks), 'frozen_unique_elements_nominal': masked_count,
        'total_unique_elements': payload['total_unique_elements'],
    }, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
