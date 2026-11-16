"""Report all fixed diagnostic items; uncertainty over source IDs, not tokens."""
from common import R, write, now, sha
from jsonl_safe import rows
import csv
import itertools
import numpy as np


def save(name, records):
    with (R / 'analysis' / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    path = R / 'analysis/math256_predictions.jsonl'
    records = rows(path)
    ids = [r['id'] for r in rows(R / 'inputs/math256.jsonl')]
    assert len(ids) == len(set(ids)) == 256
    arrays = {}
    scores = []
    for model in ['Base', 'A', 'B', 'E']:
        data = {r['id']: r for r in records if r['model'] == model}
        assert set(data) == set(ids)
        arrays[model] = np.array([data[i]['correct'] for i in ids], dtype=float)
        scores.append(dict(model=model, n=256, correct=int(arrays[model].sum()),
                           accuracy_percent=100 * arrays[model].mean(),
                           protocol='training-source diagnostic; original flexible-extract'))
    rng = np.random.default_rng(20261116)
    samples = rng.integers(256, size=(4000, 256))
    pairs = []
    for before, after in itertools.combinations(arrays, 2):
        difference = arrays[after] - arrays[before]
        low, high = np.quantile(difference[samples].mean(1), [.025, .975])
        pairs.append(dict(before=before, after=after, n=256,
                          difference_pp=100*difference.mean(), low_pp=100*low,
                          high_pp=100*high, repaired=int((difference > 0).sum()),
                          damaged=int((difference < 0).sum())))
    save('math256_scores.csv', scores)
    save('math256_paired_intervals.csv', pairs)
    write(R / 'records/MATH_SCORE_SUMMARY.json', dict(
        at=now(), input_sha256=sha(path), code_sha256=sha(__file__),
        bootstrap_unit='source question', repetitions=4000, seed=20261116,
        limitations='Single training seed; diagnostic sample uncertainty only; no score correction or subset selection.'))
    print(scores)


if __name__ == '__main__':
    main()
