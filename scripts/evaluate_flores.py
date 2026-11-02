"""Frozen 3-shot FLORES EN↔SW generation and corpus chrF++ scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from sacrebleu.metrics import CHRF
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/path/to/storage/datasets/flores200-meta-20261031/flores200_dataset')
MODEL = '/path/to/workspace/models/Qwen3-4B-Instruct-2507'
LANGS = {'eng_Latn': 'English', 'swh_Latn': 'Swahili'}


def read_lines(split: str, lang: str) -> list[str]:
    return (DATA / split / f'{lang}.{split}').read_text().splitlines()


def messages(source: str, target: str, test: str, dev_source: list[str], dev_target: list[str], example_ids: list[int]):
    def instruction(text: str) -> str:
        return f'Translate the following {LANGS[source]} text into {LANGS[target]}. Output only the translation.\n\n{text}'
    result = []
    for index in example_ids:
        result.append({'role': 'user', 'content': instruction(dev_source[index])})
        result.append({'role': 'assistant', 'content': dev_target[index]})
    result.append({'role': 'user', 'content': instruction(test)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--model-dir', type=Path, default=Path(MODEL))
    parser.add_argument('--tokenizer-dir', type=Path, default=Path(MODEL))
    parser.add_argument('--disable-thinking', action='store_true')
    parser.add_argument('--peft-adapter', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    assert args.batch_size > 0
    if args.peft_adapter and args.model_dir != Path(MODEL):
        parser.error('--peft-adapter uses the fixed Base; do not also set --model-dir')
    config = json.loads((ROOT / 'configs/flores_sw_en_500_seed42.json').read_text())
    test_ids = config['devtest_evaluation_indices_zero_based']
    example_ids = config['dev_example_indices_zero_based']
    assert len(test_ids) == len(set(test_ids)) == 500 and len(example_ids) == 3
    dev = {lang: read_lines('dev', lang) for lang in LANGS}
    test = {lang: read_lines('devtest', lang) for lang in LANGS}
    assert all(len(dev[lang]) == 997 and len(test[lang]) == 1012 for lang in LANGS)
    for lang in LANGS:
        assert not {dev[lang][i] for i in example_ids} & {test[lang][i] for i in test_ids}
    scorer = CHRF(word_order=2)
    assert scorer.corpus_score(['test'], [['test']]).score == 100.0
    if args.validate_only:
        print(json.dumps({'pairs_per_direction': 500, 'dev_shots': 3, 'chrF_word_order': 2}))
        return
    assert not args.output_dir.exists(), args.output_dir
    args.output_dir.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map='cuda:0', local_files_only=True)
    if args.peft_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.peft_adapter, is_trainable=False, local_files_only=True)
    model.eval()

    summaries = {}
    for source, target in [('eng_Latn', 'swh_Latn'), ('swh_Latn', 'eng_Latn')]:
        label = f'{source}_to_{target}'
        out_path = args.output_dir / f'{label}.jsonl'
        predictions = []
        references = []
        with out_path.open('x') as out:
            for start in range(0, len(test_ids), args.batch_size):
                batch_ids = test_ids[start:start + args.batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        messages(source, target, test[source][i], dev[source], dev[target], example_ids),
                        tokenize=False, add_generation_prompt=True,
                        **({'enable_thinking': False} if args.disable_thinking else {}),
                    ) for i in batch_ids
                ]
                encoded = tokenizer(prompts, padding=True, return_tensors='pt').to('cuda:0')
                with torch.inference_mode():
                    outputs = model.generate(
                        **encoded, do_sample=False, num_beams=1, max_new_tokens=512,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_ids = outputs[:, encoded['input_ids'].shape[1]:]
                for i, prompt, ids in zip(batch_ids, prompts, new_ids):
                    raw = tokenizer.decode(ids, skip_special_tokens=True)
                    prediction = raw.strip()
                    reference = test[target][i]
                    predictions.append(prediction)
                    references.append(reference)
                    out.write(json.dumps({
                        'devtest_index_zero_based': i,
                        'source_language': source, 'target_language': target,
                        'source': test[source][i], 'reference': reference,
                        'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                        'raw_output': raw, 'prediction': prediction,
                        'generated_token_count_padded': int(ids.numel()),
                    }, ensure_ascii=False) + '\n')
                out.flush()
                print(f'{label}: {start + len(batch_ids)}/{len(test_ids)}', flush=True)
        assert len(predictions) == len(references) == 500
        result = scorer.corpus_score(predictions, [references])
        summaries[label] = {'samples': 500, 'chrF++': result.score, 'signature': str(scorer.get_signature()), 'output': str(out_path)}
    (args.output_dir / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == '__main__':
    main()
