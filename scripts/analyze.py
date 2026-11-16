"""Recompute published task means from text-free per-item vectors (CPU only)."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    args = parser.parse_args()
    vectors = json.loads(args.vectors.read_text())
    with args.scores.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        key = row['model'] + '_' + row['task']
        values = vectors[key]['values']
        assert len(values) == int(row['n']), key
        score = sum(values) / len(values)
        assert abs(score - float(row['score_percent'])) < 1e-9, key
    print(json.dumps({'verified_score_cells': len(rows), 'model_inference': False}))


if __name__ == '__main__':
    main()
