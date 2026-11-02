"""Independently recompute full harness multiple-choice accuracy from samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def only_one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f'{label}: expected exactly one file, got {len(paths)}')
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--task-prefix', required=True)
    parser.add_argument('--expected-tasks', type=int, required=True)
    parser.add_argument('--expected-total', type=int, required=True)
    parser.add_argument('--metric', default='acc')
    args = parser.parse_args()

    result_file = only_one(list(args.run_dir.glob('results/**/results_*.json')), 'result')
    summary = json.loads(result_file.read_text())
    tasks = {
        name: result for name, result in summary['results'].items()
        if name.startswith(args.task_prefix) and 'sample_len' in result
        and list(result_file.parent.glob(f'samples_{name}_*.jsonl'))
    }
    if len(tasks) != args.expected_tasks:
        raise ValueError(f'expected {args.expected_tasks} task results, got {len(tasks)}')
    output = {'run_dir': str(args.run_dir), 'result_file': str(result_file), 'metric': args.metric, 'tasks': {}}
    correct_total = 0
    count_total = 0
    for task_name, result in sorted(tasks.items()):
        sample_file = only_one(list(result_file.parent.glob(f'samples_{task_name}_*.jsonl')), task_name)
        # JSON text can contain Unicode U+2028/U+2029 in a prompt; splitlines()
        # treats these as separators even though JSONL records end only at LF.
        with sample_file.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        ids = [row['doc_id'] for row in rows]
        if len(ids) != len(set(ids)) or len(ids) != result['sample_len']:
            raise ValueError(f'{task_name}: duplicate or missing sample')
        values = [row[args.metric] for row in rows]
        if any(value not in (0, 1, 0.0, 1.0) for value in values):
            raise ValueError(f'{task_name}: nonbinary {args.metric}')
        correct = int(sum(values))
        count = len(rows)
        reported = result[f'{args.metric},none']
        if abs(correct / count - reported) > 1e-10:
            raise ValueError(f'{task_name}: saved samples disagree with summary')
        output['tasks'][task_name] = {'count': count, 'correct': correct, 'accuracy': correct / count}
        correct_total += correct
        count_total += count
    if count_total != args.expected_total:
        raise ValueError(f'expected {args.expected_total} samples, got {count_total}')
    output.update({'total_count': count_total, 'total_correct': correct_total,
                   'micro_accuracy': correct_total / count_total, 'status': 'validated'})
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
