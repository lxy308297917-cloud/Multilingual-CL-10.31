"""Full independent eligible data, exact LF assistant mask and exposure IDs."""
import os, json, collections, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from common import *
from transformers import AutoTokenizer
from llamafactory.hparams import DataArguments
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from datasets import Dataset, DatasetDict
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--langs', nargs='+', default=list(LANGS))
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
args = DataArguments(template='qwen3', enable_thinking=False, train_on_prompt=False, cutoff_len=2048, packing=False)
template = get_template_and_fix_tokenizer(tok, args)
processor = SupervisedDatasetProcessor(template, tok, None, args)
d = DATA / 'mathoctopus-audit-v1'
audit = json.loads((d / 'audit.json').read_text())
out = DATA / 'training-v1'
out.mkdir(exist_ok=True)
info = {}
summary = {}
for lang in a.langs:
    source = d / f'{lang}_all_valid.jsonl'
    assert sha(source) == audit['languages'][lang]['records_sha256']
    rs = rows(source)
    target = out / lang
    if target.exists():
        summary[lang] = json.loads((target / 'training_data.json').read_text())
        continue
    target.mkdir()
    features = []
    ids = []
    stats = []
    fingerprints = collections.defaultdict(list)
    records = []
    for r in rs:
        msg = [{'role': 'user', 'content': r['prompt']}, {'role': 'assistant', 'content': r['answer']}]
        src, ans = template.encode_multiturn(tok, msg, None, None)[0]
        ts, ls = processor._encode_data_example(msg[:1], msg[1:], None, None, [], [], [])
        assert ts and len(ts) == len(ls) and any((x == -100 for x in ls)) and any((x != -100 for x in ls[1:]))
        features.append({'input_ids': ts, 'attention_mask': [1] * len(ts), 'labels': ls})
        fp = __import__('hashlib').sha256(json.dumps([ts, ls], separators=(',', ':')).encode()).hexdigest()
        fingerprints[fp].append(r['id'])
        ids.append(r['id'])
        stats.append([len(src) + len(ans), len(ans), len(ts), sum((x != -100 for x in ls[1:])), len(ts) < len(src) + len(ans), sum((x != -100 for x in ls[1:])) < len(ans)])
        records.append({'messages': msg})
    DatasetDict({'train': Dataset.from_list(features)}).save_to_disk(str(target / 'tokenized'))
    p = target / (lang + '_sft.jsonl')
    p.write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in records)))
    dump(target / 'dataset_info.json', {lang + '_sft': {'file_name': p.name, 'formatting': 'sharegpt', 'columns': {'messages': 'messages'}, 'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'}}})
    dump(target / 'fingerprint_source_ids.json', fingerprints)
    ar = np.asarray(stats, dtype=np.int64)
    values = {'language': lang, 'samples': len(rs), 'source_file_sha256': sha(source), 'ordered_source_ids_sha256': oh(ids), 'source_ids': ids, 'token_stats_columns': ['untruncated_input', 'untruncated_assistant', 'input', 'shifted_supervised', 'any_truncation', 'target_truncation'], 'token_totals_one_unpadded_epoch': ar.sum(0).tolist(), 'token_quantiles_0_50_90_95_99_100': np.percentile(ar, [0, 50, 90, 95, 99, 100], axis=0).tolist(), 'fingerprint_file_sha256': sha(target / 'fingerprint_source_ids.json'), 'tokenized_files_sha256': {str(p.relative_to(target)): sha(p) for p in sorted((target / 'tokenized').rglob('*')) if p.is_file()}, 'no_packing': True, 'assistant_only': True, 'mask_checks_all_records': True, 'duplicate_input_label_fingerprints': sum((len(v) - 1 for v in fingerprints.values())), 'raw_record_file_sha256': sha(p)}
    dump(target / 'training_data.json', values)
    summary[lang] = values
    print(json.dumps({k: v for k, v in values.items() if k not in ['source_ids', 'tokenized_files_sha256']}), flush=True)
dump(out / 'summary.json', summary)
