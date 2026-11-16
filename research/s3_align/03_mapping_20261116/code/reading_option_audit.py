"""Descriptive position/transition audit of saved upstream likelihood results.

Does not permute options, rerun a model, select a prompt, or replace primary scores.
"""
from common import R, LOG, write, sha, now
from jsonl_safe import rows
from pathlib import Path
from collections import Counter
import argparse
import csv
import itertools
import numpy as np


def save(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, details, sources, data, missing = [], [], {}, {}, []
    for model in ['Base', 'T0', 'T1', 'T2']:
        root = LOG / 'evaluation' / model / 'reading'
        if not (root / 'DONE.json').exists():
            missing.append(model)
            continue
        files = list((root / 'results').rglob('samples_belebele_swh_Latn_locked_*.jsonl'))
        assert len(files) == 1
        items = rows(files[0])
        assert len(items) == 900
        data[model] = {r['doc_id']: r for r in items}
        prediction_counts, target_counts, correct_by_target = Counter(), Counter(), Counter()
        for item in items:
            likelihoods = [float(r[0]) for r in item['filtered_resps']]
            assert len(likelihoods) == 4
            predicted, target = int(np.argmax(likelihoods)), int(item['target'])
            assert int(predicted == target) == item['acc'] == item['acc_norm']
            prediction_counts[predicted] += 1
            target_counts[target] += 1
            correct_by_target[target] += item['acc']
            details.append(dict(model=model, doc_id=item['doc_id'], predicted='ABCD'[predicted],
                                target='ABCD'[target], official_acc=item['acc'],
                                gold_margin=likelihoods[target] - max(v for j, v in enumerate(likelihoods) if j != target)))
        for label in range(4):
            summaries.append(dict(model=model, option='ABCD'[label], n=900,
                                  predicted_count=prediction_counts[label], target_count=target_counts[label],
                                  correct_at_target=correct_by_target[label],
                                  accuracy_when_target_percent=100*correct_by_target[label]/target_counts[label]))
        sources[str(files[0])] = sha(files[0])
    transitions = []
    for before, after in itertools.combinations(data, 2):
        assert data[before].keys() == data[after].keys()
        counts = Counter()
        for key, left in data[before].items():
            right = data[after][key]
            assert left['doc'] == right['doc'] and left['prompt_hash'] == right['prompt_hash']
            counts[{(1, 1): 'retained', (1, 0): 'lost', (0, 1): 'gained', (0, 0): 'both_wrong'}[
                int(left['acc']), int(right['acc'])]] += 1
        transitions.append(dict(before=before, after=after, n=900,
                                **{k: counts[k] for k in ['retained', 'lost', 'gained', 'both_wrong']}))
    assert args.allow_partial or not missing, missing
    save(args.output/'option_distribution.csv', summaries)
    save(args.output/'reading_items.csv', details)
    save(args.output/'reading_transitions.csv', transitions)
    write(args.output/'AUDIT.json', dict(at=now(), status='PARTIAL' if missing else 'COMPLETE',
          missing=missing, sources=sources, code_sha256=sha(__file__),
          limitations='Post-hoc descriptive option frequencies; no option permutation or causal position-bias test. Stored primary upstream scores unchanged and cross-checked.'))


if __name__ == '__main__':
    main()
