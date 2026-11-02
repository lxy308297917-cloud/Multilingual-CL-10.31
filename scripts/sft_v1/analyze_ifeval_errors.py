"""Describe saved official IFEval decisions by instruction type; no rescoring."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

LOG = Path('/path/to/workspace/logs/qwen3-multilingual/sft-v1')
TASK = 'sft_v1_ifeval'
METRICS = ['prompt_level_strict_acc', 'inst_level_strict_acc',
           'prompt_level_loose_acc', 'inst_level_loose_acc']


def read(run):
    assert (run / 'exit_code.txt').read_text().strip() == '0', run
    files = list(run.glob(f'results/**/samples_{TASK}_*.jsonl'))
    results = list(run.glob('results/**/results_*.json'))
    assert len(files) == len(results) == 1
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert len(rows) == 541
    official = json.loads(results[0].read_text())['results'][TASK]
    for key in METRICS:
        values = [v for row in rows for v in (row[key] if isinstance(row[key], list) else [row[key]])]
        assert all(type(v) is bool for v in values)
        assert len(values) == (834 if key.startswith('inst_') else 541)
        assert abs(sum(values) / len(values) - official[key + ',none']) < 1e-12
    return rows, files[0], {key: official[key + ',none'] for key in METRICS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=[42, 43, 44], default=42)
    a = p.parse_args()
    assert not a.output.exists()
    base, base_file, base_scores = read(LOG / 'base-core-other-core')
    report = {'training_seed': a.seed, 'status': 'Saved official decisions reconstructed; no inference, rescoring or checkpoint selection',
              'scope': f'Seed{a.seed} EN IFEval541 prompts/834 instructions; descriptive types may share prompts, not independent trials',
              'base_file': str(base_file), 'base_official_scores': base_scores, 'methods': {}}
    for method in ['full', 'lora', 'ssu']:
        run = LOG / (f'final-{method}-core-other' if a.seed == 42 else f'seed{a.seed}-{method}-core-other-core')
        rows, file, scores = read(run)
        types = defaultdict(Counter)
        examples = {}
        for before, after in zip(base, rows):
            assert all(before[k] == after[k] for k in ['doc_id', 'doc_hash', 'prompt_hash', 'doc'])
            tags = before['doc']['instruction_id_list']
            assert len(tags) == len(before['inst_level_strict_acc']) == len(after['inst_level_strict_acc'])
            for index, tag in enumerate(tags):
                b, m = before['inst_level_strict_acc'][index], after['inst_level_strict_acc'][index]
                c = types[tag]
                c['instructions'] += 1
                c['base_correct'] += b
                c['model_correct'] += m
                c['base_pass_model_fail'] += b and not m
                c['base_fail_model_pass'] += m and not b
                if b and not m and tag not in examples:
                    examples[tag] = {'doc_id': before['doc_id'], 'instruction_index': index,
                                     'prompt_excerpt': before['doc']['prompt'][:500],
                                     'model_response_excerpt': str(after['filtered_resps'][0])[:500]}
        assert sum(v['instructions'] for v in types.values()) == 834
        losses = {tag: dict(c, net_correct_change=c['model_correct'] - c['base_correct']) for tag, c in types.items()}
        report['methods'][method] = {'file': str(file), 'official_scores': scores,
            'types': dict(sorted(losses.items(), key=lambda item: (item[1]['net_correct_change'], item[0]))),
            'fixed_first_failure_examples_by_type': examples}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(a.output), 'status': report['status']}))


if __name__ == '__main__':
    main()
