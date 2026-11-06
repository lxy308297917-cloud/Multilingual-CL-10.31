"""Recompute reported scores and paired intervals using answer-free public files."""
from profile_paths import resolve
import argparse, json, csv, gzip, hashlib, collections
from pathlib import Path
import numpy as np
import importlib.metadata
from sacrebleu.metrics import CHRF, BLEU

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def table(p):
    with gzip.open(p, 'rt', newline='') as f:
        return list(csv.DictReader(f))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--partial', action='store_true')
    p.add_argument('--profile', choices=['formal', 'supplemental'], default='formal')
    a = p.parse_args()
    assert not a.output.exists()
    assert importlib.metadata.version('sacrebleu') == '2.6.0'
    root = a.analysis
    scores = json.loads((root / 'scores.json').read_text())
    ci = json.loads((root / 'paired_comparisons.json').read_text())
    lookup = {(x['model'], x['task'], x['metric']): x for x in scores}
    interval = {(x['model'], x['task'], x['metric']): x for x in ci}
    items = {}
    counts = 0
    for spec in json.loads((root / 'per_item_scores/manifest.json').read_text()):
        f = root / 'per_item_scores' / spec['file']
        assert sha(f) == spec['sha256']
        rows = table(f)
        assert len(rows) == len({x['item_id_sha256'] for x in rows}) == spec['rows']
        key = (spec['model'], spec['task'], spec['metric'])
        items[key] = rows
        counts += len(rows)
        factor = 1 if lookup[key]['unit'] == 'points' else 100
        score = factor * sum((float(x['numerator']) for x in rows)) / sum((float(x['denominator']) for x in rows))
        assert abs(score - lookup[key]['score']) < 1e-08, key
    stats = {}
    for spec in json.loads((root / 'corpus_statistics/manifest.json').read_text()):
        f = root / 'corpus_statistics' / spec['file']
        assert sha(f) == spec['sha256']
        rows = table(f)
        assert len(rows) == len({x['item_id_sha256'] for x in rows}) == spec['rows']
        key = (spec['model'], spec['task'], spec['metric'])
        matrix = np.asarray([json.loads(x['statistics']) for x in rows], dtype=np.int64)
        assert matrix.shape == (spec['rows'], spec['statistics_dim'])
        scorer = CHRF(char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False, eps_smoothing=False) if spec['metric'] == 'chrF++' else BLEU(lowercase=False, tokenize='13a', smooth_method='exp', smooth_value=None, effective_order=False)
        score = scorer._compute_score_from_stats(matrix.sum(axis=0).tolist()).score
        assert abs(score - lookup[key]['score']) < 1e-10
        stats[key] = (rows, matrix, scorer)
    checked = 0
    for key, expected in interval.items():
        role, task, metric = key
        basekey = ('base', task, metric)
        if key in items:
            target, base = (items[key], items[basekey])
            assert [(x['item_id_sha256'], x['identity_sha256'], x['denominator']) for x in target] == [(x['item_id_sha256'], x['identity_sha256'], x['denominator']) for x in base]
            groups = collections.defaultdict(lambda: [0.0, 0.0])
            for x, y in zip(target, base):
                g = groups[x['cluster_sha256']]
                g[0] += float(x['numerator']) - float(y['numerator'])
                g[1] += float(x['denominator'])
            arr = np.asarray(list(groups.values()))
            factor = 1 if lookup[key]['unit'] == 'points' else 100
            point = factor * arr[:, 0].sum() / arr[:, 1].sum()
            rng = np.random.default_rng(expected['seed'])
            values = []
            for _ in range(expected['repeats']):
                draw = arr[rng.integers(0, len(arr), len(arr))]
                values.append(factor * draw[:, 0].sum() / draw[:, 1].sum())
        else:
            tr, ta, scorer = stats[key]
            br, ba, _ = stats[basekey]
            assert [(x['item_id_sha256'], x['identity_sha256']) for x in tr] == [(x['item_id_sha256'], x['identity_sha256']) for x in br]
            compute = lambda z: scorer._compute_score_from_stats(z.sum(axis=0).tolist()).score
            point = compute(ta) - compute(ba)
            rng = np.random.default_rng(expected['seed'])
            values = []
            for _ in range(expected['repeats']):
                indices = rng.integers(0, len(ta), len(ta))
                values.append(compute(ta[indices]) - compute(ba[indices]))
        lo, hi = np.percentile(values, [2.5, 97.5])
        assert all((abs(float(u) - float(v)) < 1e-08 for u, v in [(point, expected['delta']), (lo, expected['ci95_low']), (hi, expected['ci95_high'])])), key
        checked += 1
    if not a.partial:
        expected = (80, 60, 64, 16) if a.profile == 'formal' else (12, 9, 12, 0)
        assert (len(scores), checked, len(items), len(stats)) == expected, ((len(scores), checked, len(items), len(stats)), expected)
    result = {'status': 'VERIFIED_PUBLIC_RESULTS_COMPLETE' if not a.partial else 'VERIFIED_AVAILABLE_PUBLIC_RESULTS_PARTIAL', 'profile': a.profile, 'scores_recomputed': len(items) + len(stats), 'intervals_recomputed': checked, 'answer_free_item_metric_rows': counts, 'item_metric_files': len(items), 'corpus_statistic_files': len(stats), 'requires_raw_questions_answers_or_outputs': False, 'requires_model_or_GPU': False, 'interval_seed_and_repeats_from_frozen_metadata': True}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
if __name__ == '__main__':
    main()
