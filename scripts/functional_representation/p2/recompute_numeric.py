"""Independent CPU recomputation from public numeric sufficient statistics."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

def bootstrap_interval(a, b, w, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    draws = []
    for start in range(0, 2000, 50):
        idx = rng.integers(0, len(a), size=(min(50, 2000 - start), len(a)))
        draws.extend((100 * np.sum(b[idx] - a[idx], axis=1) / np.sum(w[idx], axis=1)).tolist())
    return np.percentile(draws, [2.5, 97.5])

def cpu_oracles():
    a = np.array([0., 3.]);b = np.array([1., 2.]);weights = np.array([1., 3.])
    assert 100 * np.sum(b - a) / np.sum(weights) == 0
    # Two prompt clusters: all-first => +100pp, all-second => -100/3pp,
    # mixed => 0. A naive average of instruction fractions would be wrong.
    assert np.allclose(bootstrap_interval(a, b, weights, [42, 3, 2, 6]), [-100 / 3, 100])
    assert np.array_equal(bootstrap_interval(a, a, weights, [42, 3, 2, 6]), [0, 0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--allow-partial', action='store_true')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    cpu_oracles()
    data = json.loads((args.root / 'numeric_items.json').read_text())
    scores = list(csv.DictReader((args.root / 'scores.csv').open()))
    rows = list(csv.DictReader((args.root / 'paired_changes.csv').open())) if (args.root / 'paired_changes.csv').exists() else []
    absolute = {(x['model'], x['task'], x['metric']): x for x in scores}
    paired = {(x['model'], x['reference'], x['task'], x['metric']): x for x in rows}
    assert len(absolute) == len(scores) and len(paired) == len(rows)
    if not args.allow_partial:
        assert len(scores) == 40 and len(rows) == 48
    checked_scores, checked_pairs = 0, 0
    for index, record in enumerate(data['metrics']):
        models = record['models'];values = np.array(record['values'], dtype=float)
        weights = np.array(record['item_weights'], dtype=float)
        assert values.shape == (len(weights), len(models)) and np.all(np.isfinite(values))
        assert np.all(weights >= 1) and np.all(values >= 0) and np.all(values <= weights[:, None])
        if not args.allow_partial:
            assert models == data['model_order']
        for column, model in enumerate(models):
            key = (model, record['task'], record['metric'])
            observed = 100 * values[:, column].sum() / weights.sum()
            assert abs(observed - float(absolute[key]['score'])) < 1e-8
            checked_scores += 1
        for model in ['all_layer', 'full_feature', 'ce_replay']:
            for reference in ['Base'] + (['all_layer'] if model != 'all_layer' else []) + (['ce_replay'] if model == 'full_feature' else []):
                if model not in models or reference not in models:
                    continue
                key = (model, reference, record['task'], record['metric'])
                entry = paired[key]
                seed = [42, data['model_order'].index(model), data['model_order'].index(reference), index]
                assert json.loads(entry['bootstrap_seed']) == seed
                a = values[:, models.index(reference)];b = values[:, models.index(model)]
                assert abs(100 * (b - a).sum() / weights.sum() - float(entry['difference'])) < 1e-8
                lo, hi = bootstrap_interval(a, b, weights, seed)
                assert abs(lo - float(entry['ci95_low'])) < 1e-8 and abs(hi - float(entry['ci95_high'])) < 1e-8
                checked_pairs += 1
    assert checked_scores == len(scores) and checked_pairs == len(rows)
    report = dict(absolute_rows_recomputed=checked_scores, paired_intervals_recomputed=checked_pairs,
                  weighted_prompt_cluster_oracle_passed=True, complete=checked_scores == 40 and checked_pairs == 48,
                  raw_questions_or_responses_required=False, source='Numeric sufficient statistics, independent vectorized bootstrap')
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
