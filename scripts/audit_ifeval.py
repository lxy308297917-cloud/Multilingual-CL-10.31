"""Recompute official IFEval strict rates from saved harness sample scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def one(paths: list[Path]) -> Path:
    if len(paths) != 1:
        raise ValueError(f'expected one file, got {len(paths)}')
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    result_path = one(list(args.run_dir.glob('results/**/results_*.json')))
    sample_path = one(list(result_path.parent.glob('samples_ifeval_locked_*.jsonl')))
    summary = json.loads(result_path.read_text())['results']['ifeval_locked']
    with sample_path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row['doc_id'] for row in rows]
    if len(rows) != 541 or len(set(ids)) != 541 or summary['sample_len'] != 541:
        raise ValueError('expected 541 unique prompts')
    # Validate original scores before conversion: int(0.5) would hide invalid data.
    prompt_values = [row['prompt_level_strict_acc'] for row in rows]
    instruction_values = [value for row in rows for value in row['inst_level_strict_acc']]
    if len(instruction_values) != 834:
        raise ValueError(f'expected 834 instructions, got {len(instruction_values)}')
    if any(value not in (0, 1) for value in prompt_values + instruction_values):
        raise ValueError('nonbinary strict score')
    prompt_correct = int(sum(prompt_values))
    instruction_correct = int(sum(instruction_values))
    prompt_rate = prompt_correct / 541
    instruction_rate = instruction_correct / 834
    if abs(prompt_rate - summary['prompt_level_strict_acc,none']) > 1e-10:
        raise ValueError('prompt strict rate disagrees with harness')
    if abs(instruction_rate - summary['inst_level_strict_acc,none']) > 1e-10:
        raise ValueError('instruction strict rate disagrees with harness')
    print(json.dumps({
        'status': 'validated', 'run_dir': str(args.run_dir),
        'prompt_count': 541, 'prompt_correct_strict': prompt_correct,
        'prompt_strict_rate': prompt_rate, 'instruction_count': 834,
        'instruction_correct_strict': instruction_correct,
        'instruction_strict_rate': instruction_rate,
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
