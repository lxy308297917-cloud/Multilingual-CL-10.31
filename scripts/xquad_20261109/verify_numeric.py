"""Recompute the complete public XQuAD score/paired tables from numeric vectors."""
from pathlib import Path
import csv, datetime, hashlib, json
import numpy as np

def main():
    root = Path(__file__).resolve().parents[2]
    q = root / 'research/xquad_20261109'
    models = json.loads((root / 'configs/xquad_20261109/models.json').read_text())
    scores = list(csv.DictReader((q / 'scores.csv').open()))
    pairs = list(csv.DictReader((q / 'paired_changes.csv').open()))
    assert len(scores) == len(models) == 10
    assert {x['model'] for x in scores} == set(models)
    vectors = {}; inputs = []
    for row in scores:
        name = row['model']; file = q / 'numeric' / (name + '.json')
        raw = file.read_bytes(); v = json.loads(raw)
        assert v['model'] == name and v['language'] == row['language'] == models[name]['language']
        assert len(v['item_ids']) == len(set(v['item_ids'])) == 1190
        assert len(v['input_identity_sha256']) == len(v['clusters']) == 1190
        assert len(set(v['clusters'])) == int(row['paragraph_clusters']) == 240
        assert int(row['samples']) == 1190
        for metric in ['EM', 'F1']:
            a = np.asarray(v[metric], dtype=float)
            assert a.shape == (1190,) and np.isfinite(a).all()
            assert np.all((a >= 0) & (a <= 100))
            if metric == 'EM': assert np.all((a == 0) | (a == 100))
            assert abs(float(a.mean()) - float(row[metric])) < 1e-10
        vectors[name] = v
        inputs.append(dict(path=str(file.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest()))
    expected = {(a, b, metric) for i, a in enumerate(models) for b in list(models)[i+1:]
                if models[a]['language'] == models[b]['language'] for metric in ['EM', 'F1']}
    assert len(pairs) == len(expected) == 42
    observed = set()
    for row in pairs:
        key = row['reference'], row['model'], row['metric']
        assert key in expected and key not in observed; observed.add(key)
        a, b = vectors[row['reference']], vectors[row['model']]
        for field in ['item_ids', 'input_identity_sha256', 'clusters']: assert a[field] == b[field]
        assert row['language'] == a['language'] == b['language']
        assert int(row['samples']) == 1190 and int(row['paragraph_clusters']) == 240 and int(row['resamples']) == 2000
        delta = np.asarray(b[row['metric']], float) - np.asarray(a[row['metric']], float)
        clusters, inverse = np.unique(a['clusters'], return_inverse=True)
        counts = np.bincount(inverse, minlength=len(clusters))
        totals = np.bincount(inverse, weights=delta, minlength=len(clusters))
        rng = np.random.default_rng(42); draws = []
        for _ in range(40):
            index = rng.integers(len(clusters), size=(50, len(clusters)))
            draws.extend(totals[index].sum(1) / counts[index].sum(1))
        low, high = np.quantile(draws, [.025, .975])
        for field, value in [('delta', delta.mean()), ('ci95_low', low), ('ci95_high', high)]:
            assert abs(float(row[field]) - float(value)) < 1e-10, (key, field)
    assert observed == expected
    report = dict(status='PASS_COMPLETE_NUMERIC_RECOMPUTATION',
                  at=datetime.datetime.now().astimezone().isoformat(), models=10,
                  item_score_vectors=11900, absolute_metrics=20, paired_intervals=42,
                  clusters_per_language=240, resamples=2000, numpy=np.__version__,
                  scope='Numeric consistency only; original generated outputs and official scorer were separately verified by report.py against retained local raw records.',
                  inputs=inputs, table_sha256={n:hashlib.sha256((q/n).read_bytes()).hexdigest() for n in ['scores.csv', 'paired_changes.csv']})
    (q / 'numeric_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','models','item_score_vectors','absolute_metrics','paired_intervals']}))

if __name__ == '__main__': main()
