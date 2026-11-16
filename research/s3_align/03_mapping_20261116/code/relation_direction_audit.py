"""Expose original/counterfactual asymmetry without adding diagnostic queries."""
from common import R, write, now, sha
import csv
from collections import defaultdict


def main():
    source = R / 'analysis/relation_checks.csv'
    groups = defaultdict(list)
    with source.open() as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[k] for k in ['model', 'readout', 'mapping', 'method', 'family'])
            groups[key].append(row)
    results = []
    for key, records in sorted(groups.items()):
        original = {r['id']: r for r in records if r['direction'] == 'original'}
        counter = {r['id']: r for r in records if r['direction'] == 'counterfactual'}
        assert original.keys() == counter.keys()
        n = len(original)
        original_correct = sum(float(r['correct']) for r in original.values())
        counter_correct = sum(float(r['correct']) for r in counter.values())
        both = sum(float(original[i]['correct']) * float(counter[i]['correct']) for i in original)
        # Positive/negative English candidates switch roles for the second query.
        same_choice = sum((float(original[i]['margin']) > 0) != (float(counter[i]['margin']) > 0)
                          for i in original if not float(original[i]['tie']) and not float(counter[i]['tie']))
        results.append(dict(zip(['model', 'readout', 'mapping', 'method', 'family'], key)) |
                       dict(n_units=n, original_correct=original_correct,
                            counterfactual_correct=counter_correct,
                            original_accuracy=original_correct/n,
                            counterfactual_accuracy=counter_correct/n,
                            both_directions_correct=both/n,
                            unchanged_english_choice=same_choice/n))
    with (R / 'analysis/relation_direction_audit.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write(R / 'records/RELATION_DIRECTION_AUDIT.json', dict(
        at=now(), source_sha256=sha(source), code_sha256=sha(__file__),
        interpretation='Descriptive decomposition of predeclared balanced queries; no new inputs, fitting, selection, or main metric.'))


if __name__ == '__main__':
    main()
