"""Paired saved-score changes for core tasks and frozen HF choice benchmarks.

Checks actual question/prompt identity and official aggregates. Bootstrap
resamples prompts (keeping each prompt's instruction cluster together); it
never re-scores responses, selects a model, or changes the main metric.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

LOG = Path('/path/to/workspace/logs/qwen3-multilingual/sft-v1')
TASKS = ('sft_v1_gsm8k', 'sft_v1_ifeval', 'sft_v1_mgsm_native_cot_sw', 'sft_v1_xifeval_sw')


def unique(paths):
    paths = list(paths)
    assert len(paths) == 1, paths
    return paths[0]


def load(run, tasks=TASKS):
    assert (run/'exit_code.txt').read_text().strip() == '0'
    result = unique(run.glob('results/**/results_*.json'))
    summary = json.loads(result.read_text())['results']
    rows = {}
    for task in tasks:
        path = unique(result.parent.glob('samples_' + task + '_*.jsonl'))
        rows[task] = [json.loads(line) for line in path.open()]
    return summary, rows


def choice_diagnostics(rows):
    """Saved likelihood choices only; diagnostic, never changes official ACC."""
    confusion = np.zeros((4, 4), dtype=int)
    for row in rows:
        likelihoods = [float(x[0]) for x in row['filtered_resps']]
        assert len(likelihoods) == 4 and np.isfinite(likelihoods).all()
        target = row['target']
        gold = 'ABCD'.index(target) if isinstance(target,str) and target in ('A','B','C','D') else int(target)
        prediction = int(np.argmax(likelihoods))
        assert float(prediction == gold) == row['acc']
        confusion[gold, prediction] += 1
    counts = confusion.sum(axis=1)
    return {'gold_counts_A_B_C_D': counts.tolist(),
            'prediction_counts_A_B_C_D': confusion.sum(axis=0).tolist(),
            'confusion_gold_rows_prediction_columns': confusion.tolist(),
            'accuracy_by_gold_A_B_C_D': np.divide(confusion.diagonal(), counts,
                out=np.zeros(4, dtype=float), where=counts>0).tolist(),
            'scope': 'Descriptive saved-choice bias; no option reorder, prompt change or corrected score'}


def compare(run, group='other-core', base_run=None):
    hf = group.startswith('hf-')
    if hf:
        assert base_run is not None, 'Choose an explicit completed original HF Base run'
        specifications = {'hf-mmlu':('mmlu_',57,14042,'mmlu_locked'),
                          'hf-gmmlu':('global_mmlu_full_sw_',57,14042,'global_mmlu_full_sw_locked'),
                          'hf-belebele':('belebele_',2,1800,None)}
        prefix,task_count,total,aggregate_name=specifications[group]
        base_path=unique(base_run.glob('results/**/results_*.json'))
        base_data=json.loads(base_path.read_text())['results']
        tasks=sorted(name for name in base_data if name.startswith(prefix) and list(base_path.parent.glob('samples_'+name+'_*.jsonl')))
        selected=[{'task':name,'rows':base_data[name]['sample_len']} for name in tasks]
        assert len(tasks)==task_count and sum(t['rows'] for t in selected)==total
        primary_metric='acc,none'
    else:
        expected=json.loads((LOG/'evaluation_environment_preflight.json').read_text())['tasks']
        selected=[t for t in expected if (t['task'].startswith('sft_v1_mmlu_prox_'+group[-2:]+'_')
                  if group!='other-core' else not t['task'].startswith('sft_v1_mmlu_prox_'))]
        tasks=[t['task'] for t in selected]
        assert len(tasks)==(4 if group=='other-core' else 14)
        assert base_run is None, 'Core groups use their frozen vLLM Base'
        base_run=LOG/('base-core-'+group)
        aggregate_name='sft_v1_mmlu_prox_'+group[-2:] if group!='other-core' else None
        primary_metric='exact_match,custom-extract' if group!='other-core' else None
    base_summary, base_rows = load(base_run, tasks)
    summary, rows = load(run, tasks)
    rng = np.random.default_rng(42)
    report = {}
    group_before = group_after = group_count = 0
    group_boot = np.zeros(2000)
    for task in tasks:
        reference = {(r['doc_id'], r['filter']): r for r in base_rows[task]}
        grouped = defaultdict(list)
        seen = set()
        for row in rows[task]:
            key = (row['doc_id'], row['filter'])
            assert key not in seen
            seen.add(key)
            old = reference[key]
            assert row['doc'] == old['doc'] and row['arguments'] == old['arguments'], task
            grouped[row['filter']].append((old, row))
        assert seen == set(reference)
        task_report = {}
        for label, pairs in grouped.items():
            assert len(pairs) == summary[task]['sample_len'] == base_summary[task]['sample_len']
            for metric in pairs[0][1]['metrics']:
                before, after, counts = [], [], []
                for old, row in pairs:
                    a = old[metric] if isinstance(old[metric], list) else [old[metric]]
                    b = row[metric] if isinstance(row[metric], list) else [row[metric]]
                    assert len(a) == len(b) and a
                    assert all(v in (0, 1) for v in a + b)
                    before.append(sum(a)); after.append(sum(b)); counts.append(len(a))
                before, after, counts = (np.asarray(x, dtype=np.float64) for x in (before, after, counts))
                base_rate, rate = before.sum()/counts.sum(), after.sum()/counts.sum()
                key = metric + ',' + label
                assert abs(base_rate-base_summary[task][key]) < 1e-10
                assert abs(rate-summary[task][key]) < 1e-10
                changes = after-before
                draws = rng.integers(0, len(pairs), size=(2000, len(pairs)))
                boot = changes[draws].sum(axis=1)/counts[draws].sum(axis=1)*100
                if key == primary_metric:
                    group_before += int(before.sum()); group_after += int(after.sum())
                    group_count += len(pairs)
                    group_boot += changes[draws].sum(axis=1)
                task_report[key] = {'base_correct': int(before.sum()), 'model_correct': int(after.sum()),
                    'denominator': int(counts.sum()), 'base_rate': float(base_rate), 'model_rate': float(rate),
                    'delta_percentage_points': float((rate-base_rate)*100),
                    'paired_prompt_bootstrap95_percentile_pp': np.quantile(boot, [.025, .975]).tolist(),
                    'prompts_with_more_correct': int((changes>0).sum()),
                    'prompts_with_fewer_correct': int((changes<0).sum()),
                    'prompts_with_same_correct': int((changes==0).sum())}
            if not hf:
                task_report['format_'+label] = {'extraction_failures': sum(r['filtered_resps'] in (['[invalid]'], [''], [None]) for _, r in pairs),
                                           'empty_outputs': sum(not r['resps'][0][0].strip() for _, r in pairs)}
        if hf:
            task_report['choice_diagnostics'] = {'base':choice_diagnostics(base_rows[task]), 'model':choice_diagnostics(rows[task])}
        report[task] = task_report
    request_count=sum(t['rows'] for t in selected)
    meta=[] if hf else [json.loads(line) for line in (run/'completion-metadata.jsonl').open()]
    if not hf:
        assert len(meta)==request_count and len({r['request_id'] for r in meta})==request_count
    aggregate = None
    if primary_metric is not None:
        name = aggregate_name
        base_result = json.loads(unique(base_run.glob('results/**/results_*.json')).read_text())
        model_result = json.loads(unique(run.glob('results/**/results_*.json')).read_text())
        for result, correct in ((base_result, group_before), (model_result, group_after)):
            if name is not None:
                top = result.get('groups', {}).get(name, result['results'].get(name))
                assert abs(top[primary_metric]-correct/group_count) < 1e-10
        aggregate = {'base_correct': group_before, 'model_correct': group_after,
                     'denominator': group_count, 'base_rate': group_before/group_count,
                     'model_rate': group_after/group_count,
                     'delta_percentage_points': (group_after-group_before)/group_count*100,
                     'paired_subject_stratified_bootstrap95_pp': np.quantile(group_boot/group_count*100, [.025, .975]).tolist()}

    return {'status': 'validated completed '+group+' against actual Base inputs and official saved scores',
            'run': str(run), 'base_run':str(base_run), 'tasks': report, 'actual_requests':None if hf else len(meta), 'question_count':request_count, 'group_aggregate': aggregate,
            'backend_stop_reasons': dict(Counter(str(r['stop_reason']) for r in meta)),
            'backend_finish_reasons': dict(Counter(str(r['finish_reason']) for r in meta)),
            'uncertainty': 'Exploratory paired prompt-cluster bootstrap,2000 replicates, seed42, percentile95%; no multiplicity correction, no admission or selection threshold',
            'scorer_changed': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--group', choices=('other-core', 'prox-en', 'prox-sw', 'hf-belebele', 'hf-mmlu', 'hf-gmmlu'), default='other-core')
    p.add_argument('--base-run',type=Path,help='Explicit original HF Base run for a supplemental choice benchmark')
    a = p.parse_args()
    assert not a.output.exists()
    a.output.write_text(json.dumps(compare(a.run, a.group, a.base_run), indent=2)+'\n')


if __name__ == '__main__':
    main()
