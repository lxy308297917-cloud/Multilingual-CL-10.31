"""Recompute frozen FLORES chrF++ from saved per-sentence generations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sacrebleu.metrics import CHRF


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/path/to/storage/datasets/flores200-meta-20261031/flores200_dataset')


def lines(lang: str) -> list[str]:
    return (DATA / 'devtest' / f'{lang}.devtest').read_text().splitlines()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/flores_sw_en_500_seed42.json').read_text())
    expected_ids = config['devtest_evaluation_indices_zero_based']
    summary = json.loads((args.run_dir / 'results' / 'summary.json').read_text())
    source_lines = {lang: lines(lang) for lang in ('eng_Latn', 'swh_Latn')}
    scorer = CHRF(word_order=2)
    output = {'status': 'validated', 'run_dir': str(args.run_dir), 'directions': {}}
    for source, target in (('eng_Latn', 'swh_Latn'), ('swh_Latn', 'eng_Latn')):
        name = f'{source}_to_{target}'
        path = args.run_dir / 'results' / f'{name}.jsonl'
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        ids = [row['devtest_index_zero_based'] for row in rows]
        if ids != expected_ids or len(set(ids)) != 500:
            raise ValueError(f'{name}: incomplete or reordered IDs')
        for row in rows:
            index = row['devtest_index_zero_based']
            if row['source'] != source_lines[source][index] or row['reference'] != source_lines[target][index]:
                raise ValueError(f'{name}/{index}: source or reference changed')
            if row['prediction'] != row['raw_output'].strip():
                raise ValueError(f'{name}/{index}: prediction differs from raw output')
            if hashlib.sha256(row['prompt'].encode()).hexdigest() != row['prompt_sha256']:
                raise ValueError(f'{name}/{index}: prompt hash differs')
        score = scorer.corpus_score([row['prediction'] for row in rows],
                                    [[row['reference'] for row in rows]]).score
        if abs(score - summary[name]['chrF++']) > 1e-10:
            raise ValueError(f'{name}: score disagrees with saved summary')
        output['directions'][name] = {'samples': 500, 'chrF++': score,
                                      'signature': str(scorer.get_signature())}
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
