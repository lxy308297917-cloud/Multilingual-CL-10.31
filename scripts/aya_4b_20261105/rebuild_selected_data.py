"""Replay frozen source IDs and verify all6400 input/label fingerprints.

No benchmark files, model outputs, or data resampling are needed for replay.
Raw source paths are explicit CLI inputs and never downloaded implicitly.
"""
from profile_paths import resolve
import argparse, hashlib, json, gzip, tarfile, zipfile, collections, os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from llamafactory.hparams import DataArguments
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(x)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--identities', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--aya', type=Path, required=True)
    p.add_argument('--tydiqa', type=Path, required=True)
    p.add_argument('--xlsum', type=Path, required=True)
    p.add_argument('--globalvoices', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    spec = json.loads((a.identities / 'reproduction_spec.json').read_text())
    assert sha(a.identities / 'selected_source_ids.jsonl.gz') == spec['selection_file_sha256']
    assert sha(a.identities / 'input_fingerprints.jsonl.gz') == spec['fingerprints_file_sha256']
    source = {s['repository']: s for s in spec['sources']}
    for repo, path in [('CohereLabs/aya_collection_language_split', a.aya), ('google-research-datasets/tydiqa', a.tydiqa), ('csebuetnlp/xlsum', a.xlsum), ('OPUS/GlobalVoices', a.globalvoices)]:
        assert sha(path) == source[repo]['sha256'], repo
    assert sha(a.model / 'tokenizer.json') == spec['tokenizer_sha256']
    assert sha(a.model / 'tokenizer_config.json') == spec['tokenizer_config_sha256']
    with gzip.open(a.identities / 'selected_source_ids.jsonl.gz', 'rt', encoding='utf-8') as f:
        selection = [json.loads(x) for x in f]
    with gzip.open(a.identities / 'input_fingerprints.jsonl.gz', 'rt', encoding='utf-8') as f:
        fingerprints = [json.loads(x) for x in f]
    assert len(selection) == len(fingerprints) == spec['rows'] == 6400
    need = collections.defaultdict(set)
    for row in selection:
        need[row['source']].add(str(row['source_id']))
    cache = {}
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    for row in pq.read_table(a.tydiqa).to_pylist():
        if row['id'] not in need['TyDiQA goldp train']:
            continue
        answer = row['answers']['text'][0]
        context = row['context']
        start = row['answers']['answer_start'][0]
        assert context[start:start + len(answer)] == answer
        enc = tok(context, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc['input_ids']) > 1400:
            offsets = enc['offset_mapping']
            left = next((i for i, (lo, hi) in enumerate(offsets) if hi > start))
            right = next((i for i, (lo, hi) in enumerate(offsets) if lo >= start + len(answer)), len(offsets))
            lo = max(0, min(left - 600, len(offsets) - 1400))
            hi = max(lo + 1400, right)
            context = context[offsets[lo][0]:offsets[min(hi, len(offsets)) - 1][1]]
            assert answer in context
        q = 'Soma kifungu kifuatacho kisha ujibu swali kwa Kiswahili. Toa jibu fupi linalotokana na kifungu pekee.\n\nKifungu:\n' + context + '\n\nSwali: ' + row['question']
        cache['TyDiQA goldp train', row['id']] = (q, answer)
    with tarfile.open(a.xlsum) as archive:
        member = next((m for m in archive if Path(m.name).name == 'swahili_train.jsonl'))
        for line in archive.extractfile(member):
            row = json.loads(line)
            if str(row['id']) in need['XL-Sum v2.0 train']:
                cache['XL-Sum v2.0 train', str(row['id'])] = ('Fupisha makala ifuatayo kwa Kiswahili. Hifadhi mambo makuu ya ukweli na toa muhtasari mfupi tu.\n\n' + row['text'], row['summary'])
    wanted = {int(id) for s, ids in need.items() if s in ['HotpotQA (T)', 'PIQA (T)', 'Flan-GEM-wiki-lingua (T)'] for id in ids}
    pos = 0
    for batch in pq.ParquetFile(a.aya).iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            if pos in wanted:
                cache[row['dataset_name'], str(pos)] = (row['inputs'], row['targets'])
            pos += 1
    with zipfile.ZipFile(a.globalvoices) as z:
        en = z.read(next((n for n in z.namelist() if n.endswith('.en')))).decode().splitlines()
        sw = z.read(next((n for n in z.namelist() if n.endswith('.sw')))).decode().splitlines()
        assert len(en) == len(sw)
        for id in need['OPUS GlobalVoices v2018q4']:
            cache['OPUS GlobalVoices v2018q4', id] = (en[int(id)], sw[int(id)])
    args = DataArguments(template='qwen3', enable_thinking=False, train_on_prompt=False, cutoff_len=2048, packing=False)
    template = get_template_and_fix_tokenizer(tok, args)
    processor = SupervisedDatasetProcessor(template, tok, None, args)
    features = []
    training = []
    stats = collections.Counter()
    input_count = label_count = 0
    for i, (row, expected) in enumerate(zip(selection, fingerprints)):
        q, answer = cache[row['source'], str(row['source_id'])]
        if row['task'] == 'en_to_sw':
            q = 'Translate the following English text into Swahili. Output only the translation.\n\n' + q
        elif row['task'] == 'sw_to_en':
            q, answer = ('Translate the following Swahili text into English. Output only the translation.\n\n' + answer, q)
        q, answer = (q.strip(), answer.strip())
        messages = [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': answer}]
        ids, labels = processor._encode_data_example(messages[:1], messages[1:], None, None, [], [], [])
        digest = hashlib.sha256(json.dumps([ids, labels], separators=(',', ':')).encode()).hexdigest()
        assert digest == expected['input_label_sha256'], (i, row['source'], row['source_id'])
        assert i == expected['position']
        assert len(ids) == expected['input_tokens']
        count = sum((x != -100 for x in labels[1:]))
        assert count == expected['shifted_assistant_tokens']
        input_count += len(ids)
        label_count += count
        stats[row['task']] += 1
        features.append({'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels})
        training.append({'instruction': q, 'output': answer})
    assert (input_count, label_count) == (spec['input_tokens'], spec['shifted_assistant_tokens'])
    a.output.mkdir(parents=True)
    with (a.output / 'train6400.jsonl').open('x') as f:
        for row in training:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (a.output / 'train6400.provenance.jsonl').write_bytes(gzip.decompress((a.identities / 'selected_source_ids.jsonl.gz').read_bytes()))
    assert sha(a.output / 'train6400.jsonl') == spec['training_jsonl_sha256']
    DatasetDict({'train': Dataset.from_list(features)}).save_to_disk(str(a.output / 'train6400_tokenized'))
    (a.output / 'dataset_info.json').write_text(json.dumps({'train6400': {'file_name': 'train6400.jsonl', 'columns': {'prompt': 'instruction', 'response': 'output'}}}, indent=2) + '\n')
    for name, digest in spec['training_files_sha256'].items():
        assert sha(a.output / name) == digest, name
    result = {'status': 'VERIFIED_EXACT_PUBLIC_REBUILD', 'rows': len(training), 'input_tokens': input_count, 'shifted_assistant_tokens': label_count, 'task_counts': dict(stats), 'input_label_fingerprints_matched': len(training), 'training_jsonl_sha256': sha(a.output / 'train6400.jsonl'), 'raw_questions_or_answers_required_in_GitHub': False, 'files': {str(p.relative_to(a.output)): sha(p) for p in a.output.rglob('*') if p.is_file()}}
    (a.output / 'REBUILD_VERIFIED.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))
if __name__ == '__main__':
    main()
