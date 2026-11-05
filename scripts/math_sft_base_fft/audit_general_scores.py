"""Independent full-sample count and arithmetic checks for official general tasks."""
import argparse, json, collections, math
from pathlib import Path
from common import *
ap = argparse.ArgumentParser()
ap.add_argument('--input', type=Path, required=True)
a = ap.parse_args()
assert (a.input / 'GPU_PHASE_DONE.json').exists()
sc = json.loads((a.input / 'scores.json').read_text())
out = {}
files = list((a.input / 'mmlu/results').glob('**/samples_*.jsonl'))
assert len(files) == 57, len(files)
n = correct = 0
digests = {}
for f in files:
    xs = rows(f)
    assert len({r['doc_id'] for r in xs}) == len(xs)
    for r in xs:
        assert len(r['resps']) == 4 and r['filter'] == 'none'
        assert all((math.isfinite(float(r['resps'][i][0][0])) for i in range(4))), ('nonfinite candidate readout', r['doc_id'])
        pred = max(range(4), key=lambda i: float(r['resps'][i][0][0]))
        hit = int(pred == int(r['target']))
        assert hit == int(r['acc'])
        correct += hit
        n += 1
    digests[str(f.relative_to(a.input))] = sha(f)
assert n == 14042, n
rate = correct / n
assert abs(rate - sc['mmlu']['mmlu_locked']['acc,none']) < 1e-10
out['MMLU'] = {'n': n, 'correct': correct, 'accuracy': rate, 'candidate_likelihood_recomputed': True, 'nonfinite_or_wrong_candidate_count': 0}
for task, count in [('sft_v1_gsm8k', 1319), ('sft_v1_ifeval', 541)]:
    files = list((a.input / 'core/results').glob('**/samples_' + task + '_*.jsonl'))
    assert len(files) == 1
    grouped = collections.defaultdict(list)
    for x in rows(files[0]):
        grouped[x['filter']].append(x)
    metrics = {}
    for filt, xs in grouped.items():
        assert len(xs) == len({x['doc_id'] for x in xs}) == count
        for key in xs[0]['metrics']:
            flags = [v for x in xs for v in (x[key] if isinstance(x[key], list) else [x[key]])]
            assert all((v in [0, 1] for v in flags))
            rate = sum(flags) / len(flags)
            assert abs(rate - sc['core'][task][key + ',' + filt]) < 1e-10
            if task == 'sft_v1_ifeval':
                assert len(flags) == (834 if key.startswith('inst_') else 541)
            metrics[key + ',' + filt] = {'n': len(flags), 'correct': sum(flags), 'score': rate}
    if task == 'sft_v1_gsm8k':
        parsed = [x['filtered_resps'][0] for x in grouped['flexible-extract']]
        valid = sum((v is not None and str(v).strip() not in ['', '[invalid]'] for v in parsed))
        out['GSM8K_answer_parsing'] = {'n': len(parsed), 'parsed': valid, 'unparsed': len(parsed) - valid, 'parse_rate': valid / len(parsed), 'rule': 'Actual official flexible-extract filter, including invalid placeholders in the full denominator.'}
    out[task] = metrics
    digests[str(files[0].relative_to(a.input))] = sha(files[0])
code = a.input / 'code-execution'
if (code / 'DONE.json').exists():
    xs = json.loads((code / 'model_evaluation-results.json').read_text())
    assert len(xs) == len({x['task_id'] for x in xs}) == 164
    assert all((x['infrastructure_status'] == 'COMPLETE' for x in xs))
    scores = json.loads((code / 'scores.json').read_text())
    assert abs(sum((x['plus_pass'] for x in xs)) / 164 - scores['pass_at_1_plus']) < 1e-10
    assert all((not x['plus_pass'] or x['base_pass'] for x in xs))
    out['HumanEval_plus'] = {'n': 164, 'correct': sum((x['plus_pass'] for x in xs)), 'actual_sandbox_execution': True}
else:
    out['HumanEval_plus'] = {'status': 'PENDING_CPU_EXECUTION'}
dump(a.input / 'independent_score_audit.json', {'status': 'COMPLETE' if (code / 'DONE.json').exists() else 'GPU_SCORES_VERIFIED_CODE_PENDING', 'metrics': out, 'sample_files_sha256': digests})
print(json.dumps(out, indent=2))
