"""Describe saved FLORES repetition without changing generations or scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

LOG = Path('/path/to/workspace/logs/qwen3-multilingual')
DIRECTIONS = ('eng_Latn_to_swh_Latn', 'swh_Latn_to_eng_Latn')


def repetition(text):
    words = text.casefold().split()
    best = {'consecutive_repeats': 0, 'repeated_words': 0, 'phrase': [], 'start_word': None}
    for width in range(1, 6):
        for offset in range(width):
            previous, count = None, 0
            for position in range(offset, len(words) - width + 1, width):
                phrase = tuple(words[position:position + width])
                count = count + 1 if phrase == previous else 1
                previous = phrase
                covered = count * width
                if count >= 10 and covered > best['repeated_words']:
                    best = dict(consecutive_repeats=count, repeated_words=covered, phrase=list(phrase),
                                start_word=position - (count - 1) * width)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = dict(scope='Exploratory description of saved complete FLORES outputs; original chrF++ retained',
        training_seed=args.seed, rule='Casefold and split whitespace; at least10 consecutive identical blocks of1–5 words',
        caveats=['Rule fixed after inspecting repetition examples, before aggregate counts; not preregistered',
                 'Generated_token_count_padded is a batch width, not each response length or truncation indicator',
                 'Repetition is observed output behavior; it does not identify a training/backend/prompt root cause'],
        methods={})
    base_inputs = {}
    for method in ['base', 'full', 'lora', 'ssu']:
        if method == 'base':
            run = LOG / 'evaluation/base17b_flores_20261031_r2'
        elif args.seed == 42:
            run = LOG / f'sft-v1/complementary-{method}-flores-v1'
        else:
            run = LOG / f'sft-v1/seed{args.seed}-{method}-hf-flores'
        assert (run / 'exit_code.txt').read_text().strip() == '0', run
        summary = json.loads((run / 'results/summary.json').read_text())
        directions = {}
        for direction in DIRECTIONS:
            path = run / 'results' / (direction + '.jsonl')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == summary[direction]['samples'] == 500
            keys = ['devtest_index_zero_based', 'source_language', 'target_language', 'source',
                    'reference', 'prompt', 'prompt_sha256']
            inputs = [{k: row[k] for k in keys} for row in rows]
            if method == 'base':
                base_inputs[direction] = inputs
            else:
                assert inputs == base_inputs[direction]
            details, examples = [], []
            lengths = []
            reference_repeats = 0
            for row in rows:
                text = row['prediction']
                assert text == row['raw_output'].strip()
                repeat = repetition(text)
                lengths.append(len(text.split()))
                reference_repeats += repetition(row['reference'])['consecutive_repeats'] >= 10
                details.append(dict(index=row['devtest_index_zero_based'], words=lengths[-1], **repeat))
                if repeat['consecutive_repeats'] >= 10 and len(examples) < 3:
                    examples.append(dict(index=row['devtest_index_zero_based'], source=row['source'],
                        reference=row['reference'], prediction=text, repetition=repeat))
            batches = [rows[i:i + 4] for i in range(0, 500, 4)]
            assert all(len({x['generated_token_count_padded'] for x in batch}) == 1 for batch in batches)
            directions[direction] = dict(samples=500, official_chrFpp=summary[direction]['chrF++'],
                repeated_response_count=sum(x['consecutive_repeats'] >= 10 for x in details),
                references_with_same_pattern=reference_repeats, empty_responses=sum(not row['prediction'] for row in rows),
                exact_source_copies=sum(row['prediction'].strip() == row['source'].strip() for row in rows),
                mean_whitespace_words=statistics.mean(lengths), median_whitespace_words=statistics.median(lengths),
                batch_width_counts=dict(Counter(batch[0]['generated_token_count_padded'] for batch in batches)),
                source_file=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                examples=examples, per_response=details)
        result['methods'][method] = directions
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({method: {direction: {'repeated': stats['repeated_response_count'],
        'samples': stats['samples'], 'chrF++': stats['official_chrFpp'],
        'batches_with_width512': stats['batch_width_counts'].get(512, 0)}
        for direction, stats in directions.items()} for method, directions in result['methods'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
