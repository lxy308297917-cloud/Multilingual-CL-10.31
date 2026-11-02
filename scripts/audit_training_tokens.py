"""Count Qwen3 CPT tokens after conservative benchmark exclusions.

Uses LLaMA-Factory's PT semantics: text + tokenizer EOS, no added special
tokens, then packing in 1,000-document map batches at the chosen cutoff.
This is a corpus budget audit, not a training run or a downstream metric.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
ASSETS = json.loads((ROOT / 'assets.lock.json').read_text())['assets']
MODEL = '/path/to/workspace/models/Qwen3-4B-Instruct-2507'


def documents(name: str):
    path = Path(ASSETS[name]['path'])
    if name == 'madlad_sw_clean':
        with gzip.open(path, 'rt', encoding='utf-8') as fh:
            for row in fh:
                yield json.loads(row)['text']
    else:
        file = pq.ParquetFile(path)
        for batch in file.iter_batches(batch_size=1024, columns=['text']):
            yield from batch.column(0).to_pylist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('asset', choices=['madlad_sw_clean', 'fineweb2_swh_train'])
    parser.add_argument('--exclusions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cutoff', type=int, default=2048)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    audit = json.loads(args.exclusions.read_text())
    assert audit['corpus'] == args.asset
    excluded = set(audit['excluded_zero_based_indices'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert not getattr(tokenizer, 'add_bos_token', False)
    eos = tokenizer.eos_token
    counts = {'source_rows': 0, 'excluded_rows': 0, 'included_rows': 0, 'raw_tokens': 0, 'packed_blocks': 0, 'dropped_tail_tokens': 0}
    batch = []
    map_batch_tokens = 0
    map_batch_rows = 0

    def tokenize_batch():
        nonlocal batch, map_batch_tokens, map_batch_rows
        if not batch:
            return
        ids = tokenizer([text + eos for text in batch], add_special_tokens=False)['input_ids']
        for row in ids:
            length = len(row)
            counts['raw_tokens'] += length
            map_batch_tokens += length
            map_batch_rows += 1
            if map_batch_rows == 1000:
                blocks, tail = divmod(map_batch_tokens, args.cutoff)
                counts['packed_blocks'] += blocks
                counts['dropped_tail_tokens'] += tail
                map_batch_tokens = map_batch_rows = 0
        batch = []

    for index, text in enumerate(documents(args.asset)):
        counts['source_rows'] += 1
        if index in excluded:
            counts['excluded_rows'] += 1
        else:
            counts['included_rows'] += 1
            batch.append(text)
            if len(batch) == 256:
                tokenize_batch()
        if args.limit and counts['source_rows'] >= args.limit:
            break
        if counts['source_rows'] % 50000 == 0:
            print(json.dumps(counts), file=sys.stderr, flush=True)
    tokenize_batch()
    if map_batch_rows:
        blocks, tail = divmod(map_batch_tokens, args.cutoff)
        counts['packed_blocks'] += blocks
        counts['dropped_tail_tokens'] += tail
    if not args.limit:
        assert counts['source_rows'] == ASSETS[args.asset]['rows']
        assert counts['excluded_rows'] == len(excluded)
    assert counts['raw_tokens'] == counts['packed_blocks'] * args.cutoff + counts['dropped_tail_tokens']
    report = {
        'asset': args.asset,
        'source_sha256': ASSETS[args.asset]['sha256'],
        'exclusions_sha256': hashlib.sha256(args.exclusions.read_bytes()).hexdigest(),
        'tokenizer_json_sha256': ASSETS['qwen3_4b_2507/tokenizer.json']['sha256'],
        'cutoff': args.cutoff,
        'map_batch_size': 1000,
        'limit': args.limit,
        **counts,
        'packed_training_tokens': counts['packed_blocks'] * args.cutoff,
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
