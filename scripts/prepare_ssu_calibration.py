"""Prepare pinned English source calibration in the SSU authors' packing style.

The source revision and sampled IDs are saved. Qwen3 tokenization is required;
the original OLMo token IDs cannot be reused for this architecture adaptation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from itertools import islice
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


SOURCE = 'allenai/tulu-3-sft-olmo-2-mixture'
REVISION = '2d335ef93d9ad35e718e201c2aabefa08032a3d3'
MODEL = '/path/to/workspace/models/Qwen3-4B-Instruct-2507'
TOKENIZER_SHA256 = 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
RAW_COUNT = 2000
SHUFFLE_BUFFER = 20000
BLOCK_SIZE = 2048
SEED = 42


def format_record(tokenizer, record):
    messages = record.get('messages') or record.get('conversations')
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False), False
        except Exception:
            return '\n\n'.join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages), True
    raise ValueError(f'Unexpected source schema for id={record.get("id")}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--selected-raw', type=Path, help='Replay a locked 2,000-row raw JSONL without network streaming')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if args.selected_raw:
        records = [json.loads(line) for line in args.selected_raw.read_text().splitlines()]
    else:
        source = load_dataset(SOURCE, revision=REVISION, split='train', streaming=True)
        source = source.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)
        records = list(islice(source, RAW_COUNT))
    if len(records) != RAW_COUNT:
        raise RuntimeError(f'Only {len(records)} source rows available')

    raw_path = args.output_dir / 'selected_raw.jsonl'
    if args.selected_raw:
        shutil.copyfile(args.selected_raw, raw_path)
    else:
        with raw_path.open('w') as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + '\n')
    ids = [record['id'] for record in records]
    if len(set(ids)) != RAW_COUNT:
        raise RuntimeError('Source sample has duplicate IDs')

    per_record_tokens = []
    fallback_ids = []
    truncated = 0
    for record in records:
        text, fallback = format_record(tokenizer, record)
        if fallback:
            fallback_ids.append(record['id'])
        full = tokenizer.encode(text, add_special_tokens=True)
        truncated += len(full) > BLOCK_SIZE
        per_record_tokens.append(full[:BLOCK_SIZE])
    blocks = []
    dropped_tail = 0
    # The authors' Dataset.map(..., batched=True) defaults to batches of 1,000
    # raw examples, so packing drops one tail per batch, not just one globally.
    for start in range(0, len(per_record_tokens), 1000):
        tokens = [token for row in per_record_tokens[start:start + 1000] for token in row]
        usable = len(tokens) // BLOCK_SIZE * BLOCK_SIZE
        blocks.extend(tokens[i:i + BLOCK_SIZE] for i in range(0, usable, BLOCK_SIZE))
        dropped_tail += len(tokens) - usable
    packed = Dataset.from_dict({
        'input_ids': blocks,
        'attention_mask': [[1] * BLOCK_SIZE for _ in blocks],
        'labels': [block.copy() for block in blocks],
    })
    packed.save_to_disk(str(args.output_dir / 'packed'))
    report = {
        'source': SOURCE, 'revision': REVISION, 'seed': SEED,
        'shuffle_buffer': SHUFFLE_BUFFER, 'raw_count': RAW_COUNT,
        'source_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'raw_jsonl_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'tokenizer': MODEL, 'tokenizer_json_sha256': TOKENIZER_SHA256,
        'block_size': BLOCK_SIZE, 'truncated_raw_rows': truncated,
        'packed_blocks': len(blocks), 'dropped_tail_tokens': dropped_tail,
        'fallback_ids': fallback_ids,
        'selected_raw_input': str(args.selected_raw) if args.selected_raw else None,
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
