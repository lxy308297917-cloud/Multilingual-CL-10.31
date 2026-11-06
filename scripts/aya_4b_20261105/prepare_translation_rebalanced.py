"""One task-rebalanced6400-example candidate; never overwrite the original mixture."""
from profile_paths import resolve
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json, hashlib, collections, zipfile, unicodedata, argparse
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from llamafactory.hparams import DataArguments
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
D = Path(resolve('DATA'))
M = Path(resolve('MODEL'))
SEED = 'aya-4b-task-aligned-v1-seed42'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def hs(s):
    return hashlib.sha256(s.encode()).hexdigest()

def norm(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())

def priority(s):
    return hs(SEED + '|' + s)

def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(exist_ok=False)
    base = D / 'mixture-v1'
    original = [json.loads(x) for x in (base / 'train6400.jsonl').open()]
    prov = [json.loads(x) for x in (base / 'train6400.provenance.jsonl').open()]
    assert len(original) == len(prov) == 6400
    grouped = collections.defaultdict(list)
    for x, y in zip(original, prov):
        grouped[y['task']].append({**x, **y})
    tok = AutoTokenizer.from_pretrained(M, local_files_only=True)
    args = DataArguments(template='qwen3', enable_thinking=False, train_on_prompt=False, cutoff_len=2048, packing=False)
    template = get_template_and_fix_tokenizer(tok, args)
    processor = SupervisedDatasetProcessor(template, tok, None, args)

    def encode(row):
        msgs = [{'role': 'user', 'content': row['instruction']}, {'role': 'assistant', 'content': row['output']}]
        ids, labels = processor._encode_data_example(msgs[:1], msgs[1:], None, None, [], [], [])
        if len(ids) > 2048:
            return None
        first = next((i for i, x in enumerate(labels) if x != -100))
        assert ids[:first] == tok.apply_chat_template(msgs[:1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        assert labels[-2:] == [151645, 198]
        return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}
    train = []
    features = []
    seen = set()
    stats = collections.defaultdict(lambda: {'rows': 0, 'input_tokens': 0, 'assistant_tokens': 0})

    def append(row, feature):
        assert row['instruction'].strip() and row['output'].strip()
        key = hs(norm(row['instruction']))
        assert key not in seen
        seen.add(key)
        train.append(row)
        features.append(feature)
        s = stats[row['task']]
        s['rows'] += 1
        s['input_tokens'] += len(feature['input_ids'])
        s['assistant_tokens'] += sum((x != -100 for x in feature['labels'][1:]))
    for task, n in [('reading', 1280), ('summary_native', 640), ('knowledge', 640), ('commonsense', 640)]:
        selected = sorted(grouped[task], key=lambda x: priority(x['source'] + '|' + x['source_id']))[:n]
        assert len(selected) == n
        for row in selected:
            feature = encode(row)
            assert feature is not None
            append(row, feature)
    excluded = set()
    benchmark = Path(resolve('BENCH_DATA') + '/prepared-v1')
    for directory in benchmark.iterdir():
        if not (directory / 'test.jsonl').is_file():
            continue
        for x in map(json.loads, (directory / 'test.jsonl').open()):
            for key in ['source', 'reference']:
                if isinstance(x.get(key), str) and x[key].strip():
                    excluded.add(hs(norm(x[key])))
    zpath = D / 'raw/globalvoices-en-sw-v2018q4.zip'
    with zipfile.ZipFile(zpath) as z:
        en = z.read(next((n for n in z.namelist() if n.endswith('.en')))).decode().splitlines()
        sw = z.read(next((n for n in z.namelist() if n.endswith('.sw')))).decode().splitlines()
        assert len(en) == len(sw)
        pool = []
        for i, (e, s) in enumerate(zip(en, sw)):
            if not 4 <= len(e.split()) <= 100 or not 4 <= len(s.split()) <= 120 or '�' in e + s or ('<unk>' in e + s):
                continue
            if hs(norm(e)) in excluded or hs(norm(s)) in excluded:
                continue
            group = hs(norm(e) + '|' + norm(s))
            if int(group[:8], 16) % 10 == 0:
                continue
            pool.append((i, e.strip(), s.strip(), group))
    selected_groups = []
    seen_parallel_sources = set()
    for i, e, s, group in sorted(pool, key=lambda x: priority('OPUS GlobalVoices v2018q4|' + str(x[0]))):
        if len(selected_groups) == 1600:
            break
        if hs(norm(e)) in seen_parallel_sources:
            continue
        common = {'source': 'OPUS GlobalVoices v2018q4', 'source_id': str(i), 'parallel_group_sha256': group}
        rs = [{**common, 'task': 'en_to_sw', 'instruction': 'Translate the following English text into Swahili. Output only the translation.\n\n' + e, 'output': s}, {**common, 'task': 'sw_to_en', 'instruction': 'Translate the following Swahili text into English. Output only the translation.\n\n' + s, 'output': e}]
        fs = [encode(r) for r in rs]
        if any((f is None for f in fs)):
            continue
        if any((hs(norm(r['instruction'])) in seen for r in rs)):
            continue
        seen_parallel_sources.add(hs(norm(e)))
        selected_groups.append(group)
        for r, f in zip(rs, fs):
            append(r, f)
    assert len(selected_groups) == 1600 and len(train) == 6400
    oldgroups = {r['parallel_group_sha256'] for r in grouped['en_to_sw']}
    assert len(oldgroups) == 640 and oldgroups <= set(selected_groups)
    order = sorted(range(len(train)), key=lambda i: priority(train[i]['task'] + '|' + train[i]['source'] + '|' + train[i]['source_id']))
    train = [train[i] for i in order]
    features = [features[i] for i in order]
    with (out / 'train6400.jsonl').open('x') as f, (out / 'train6400.provenance.jsonl').open('x') as g:
        for row in train:
            f.write(json.dumps({k: row[k] for k in ['instruction', 'output']}, ensure_ascii=False) + '\n')
            g.write(json.dumps({k: v for k, v in row.items() if k not in ['instruction', 'output']}, ensure_ascii=False) + '\n')
    DatasetDict({'train': Dataset.from_list(features)}).save_to_disk(str(out / 'train6400_tokenized'))
    dump(out / 'dataset_info.json', {'train6400': {'file_name': 'train6400.jsonl', 'columns': {'prompt': 'instruction', 'response': 'output'}}})
    manifest = {'status': 'FROZEN_DATA_CANDIDATE_NOT_TRAINED', 'rows': 6400, 'model_revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'selection_seed': SEED, 'parent_data_manifest_sha256': sha(base / 'manifest.json'), 'task_stats': dict(stats), 'input_tokens': sum((x['input_tokens'] for x in stats.values())), 'shifted_assistant_tokens': sum((x['assistant_tokens'] for x in stats.values())), 'translation_unique_parallel_groups': 1600, 'translation_rows': 3200, 'original640parallelgroups_retained': True, 'removed_task': 'summary_instructional/AyaWikiLingua', 'unchanged_source_rows': 'Reading/news subsets of parent; all640HotpotQA and640PIQA retained; nativeQA contextcrop inherited, no test-score selection', 'source_parallel_zip_sha256': sha(zpath), 'tokenizer_sha256': sha(M / 'tokenizer.json'), 'template_config_sha256': sha(M / 'tokenizer_config.json'), 'full_6400_nonthinking_prefix_and_im_end_labels_verified': True, 'truncated_selected_examples': 0, 'limitations': ['Same6400examples/400steps/effectivebatch16, but input/supervisedtokenbudgets differ fromv1', 'Token share is not gradient contribution; recipe change tests a joint rebalancing hypothesis, not a pure one-variable gradient mechanism', 'Knowledge source is unchanged and not57subject MMLU training; no promise of knowledge gain', 'Translation heldoutgroup rule remains10% and both directions stay in one split'], 'files': {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}}
    dump(out / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k in ['rows', 'task_stats', 'input_tokens', 'shifted_assistant_tokens', 'translation_unique_parallel_groups']}, ensure_ascii=False))
if __name__ == '__main__':
    main()
