"""Create independent, ID-locked train/dev/test assets from publisher releases.

Only exact identity checks; no score-dependent filtering or semantic search.
Use --track to isolate failures; existing prepared assets are immutable.
"""
from runtime import runtime_path
import argparse, collections, csv, gzip, io, json, os, re, tarfile, zipfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import pyarrow.parquet as pq
from common import DATA, MODEL, LANGS, sha, oh, dump, write_rows, norm, order, number, prompt
SRC = DATA / 'sources'

def record(track, lang, split, id, text, answer, **extra):
    return dict(track=track, lang=lang, split=split, id=str(id), text=text, answer=answer, **extra)

def massive():
    rs = []
    p = SRC / 'AmazonScience--massive/amazon-massive-dataset-1.1.tar.gz'
    with tarfile.open(p) as tf:
        for lang, locale in zip(LANGS, ['es-ES', 'th-TH', 'fr-FR', 'en-US']):
            mem = next((x for x in tf.getmembers() if x.name.endswith('/' + locale + '.jsonl')))
            for line in tf.extractfile(mem):
                x = json.loads(line)
                rs.append(record('massive', lang, x['partition'], x['id'], x['utt'], x['intent']))
    return (rs, dict(source_sha256=sha(p), alignment='Official id maps back to original SLURP semantic item; labels and partitions checked across languages.', license='CC-BY-4.0'))

def sib200():
    rs = []
    for lang, code in zip(LANGS, ['spa_Latn', 'tha_Thai', 'fra_Latn', 'eng_Latn']):
        for split, f in [('train', 'train.tsv'), ('dev', 'dev.tsv'), ('test', 'test.tsv')]:
            p = SRC / 'Davlan--sib200' / 'data' / code / f
            for x in csv.DictReader(p.open(), delimiter='\t'):
                rs.append(record('sib200', lang, split, x['index_id'], x['text'], x['category']))
    return (rs, dict(alignment='Official index_id, categories checked across languages.', license='CC-BY-SA-4.0'))

def xnli():
    rs = []
    source_counts = {}
    train_labels = collections.defaultdict(set)
    train_first = {}
    for split, raw in [('train', 'train'), ('dev', 'validation'), ('test', 'test')]:
        count = 0
        for p in sorted((SRC / 'facebook--xnli/all_languages').glob(raw + '-*.parquet')):
            for batch in pq.ParquetFile(p).iter_batches(batch_size=4096):
                for x in batch.to_pylist():
                    count += 1
                    pms = x['premise']
                    hp = x['hypothesis']
                    hps = dict(zip(hp['language'], hp['translation']))
                    semantic = oh([norm(pms['en']), norm(hps['en'])])
                    sid = f'{split}:{count - 1}:{semantic}'
                    label = ['entailment', 'neutral', 'contradiction'][x['label']]
                    if split == 'train':
                        train_labels[semantic].add(label)
                        train_first.setdefault(semantic, sid)
                    for lang in LANGS:
                        fields = {'en': ['Premise', 'Hypothesis'], 'es': ['Premisa', 'Hipótesis'], 'fr': ['Prémisse', 'Hypothèse'], 'th': ['ข้อความตั้งต้น', 'สมมติฐาน']}[lang]
                        text = f'{fields[0]}: {pms[lang]}\n{fields[1]}: {hps[lang]}'
                        rs.append(record('xnli', lang, split, sid, text, label, semantic_group=semantic, premise=pms[lang], hypothesis=hps[lang]))
        assert count, ('Missing XNLI split', split)
        source_counts[split] = count
    conflicts = {s for s, v in train_labels.items() if len(v) > 1}
    eval_groups = {r['semantic_group'] for r in rs if r['split'] != 'train'}
    excluded_groups = conflicts | eval_groups
    before = len(rs)
    rs = [r for r in rs if r['split'] != 'train' or (r['semantic_group'] not in excluded_groups and r['id'] == train_first[r['semantic_group']])]
    return (rs, dict(alignment='Shared multilingual object in official all_languages parquet. Unique annotation ID = split + joint-object row + hash(English premise,hypothesis); semantic group=English text hash. No independent language-row zipping. Official dev/test annotations retained, even repeated texts.', source_counts=source_counts, train_conflicting_semantic_groups_removed=len(conflicts), train_duplicate_extra_annotations=source_counts['train'] - len(train_labels), train_eval_semantic_groups=len(set(train_labels) & eval_groups), removed_train_language_records=before - len(rs), license='CC-BY-NC-4.0 (XNLI author release; confirm license text in source documentation)', train_translation='Machine translated MultiNLI train; dev/test human-translated XNLI.'))

def mathoctopus():
    path = SRC / 'Mathoctopus--GSM8KInstruct_Parallel/MGSM8KInstruct_Parallel.json'
    raw = [json.loads(line) for line in path.open() if line.strip()]
    gsm = Path(runtime_path('gsm8k_source', ''))
    ref = pq.read_table(gsm / 'train-00000-of-00001.parquet').to_pylist()
    canonical = {norm(x['question']): x for x in ref}
    assert len(canonical) == len(ref)
    names = {'Spanish': 'es', 'Thai': 'th', 'French': 'fr', 'English': 'en'}
    groups = []
    group = None
    counts = collections.Counter()
    rejected = collections.Counter()
    for idx, x in enumerate(raw):
        match = re.search('request in (\\w+)\\.', x['prompt'])
        assert match, x['prompt'][:150]
        language = match.group(1)
        counts[language] += 1
        question = x['prompt'].split('### Instruction:', 1)[1].rsplit('### Response:', 1)[0].strip()
        if language == 'English':
            if group:
                groups.append(group)
            c = canonical.get(norm(question))
            assert c is not None, ('English anchor not GSM8K train', idx)
            group = {'source_id': oh(norm(question)), 'source_question': question, 'gold': number(c['answer'].split('####')[-1]), 'items': {}, 'raw_start': idx}
        assert group is not None
        if language in names:
            lang = names[language]
            assert lang not in group['items'], ('Duplicate language inside explicit English-led block', idx)
            group['items'][lang] = {'question': question, 'response': x['chosen'].strip(), 'final': number(x['chosen'], lang)}
    groups.append(group)
    assert len(groups) == len(ref), (len(groups), len(ref))
    rs = []
    selected = []
    for g in groups:
        if set(g['items']) != set(LANGS):
            rejected['missing_language'] += 1
            continue
        if any((x['final'] != g['gold'] for x in g['items'].values())):
            rejected['final_numeric_disagreement_or_unparseable'] += 1
            continue
        selected.append(g)
    dev_ids = set(order([g['source_id'] for g in selected], 'math-internal-dev')[:256])
    for g in selected:
        for lang, x in g['items'].items():
            ans = x['response']
            if not re.search('####\\s*[-+]?\\d', ans):
                ans += '\n#### ' + format(__import__('decimal').Decimal(g['gold']), 'f')
            rs.append(record('mathoctopus', lang, 'dev' if g['source_id'] in dev_ids else 'train', g['source_id'], x['question'], ans, gold_numeric=g['gold']))
    for lang, name in zip(LANGS, ['Spanish', 'Thai', 'French', 'English']):
        test_rows = [json.loads(line) for line in (SRC / f'Mathoctopus--MSVAMP/test_{name}.json').open() if line.strip()]
        for i, x in enumerate(test_rows):
            text = x.get('m_query', x.get('question', x.get('Question', x.get('Body', ''))))
            if 'Body' in x and 'Question' in x:
                text = x['Body'] + ' ' + x['Question']
            answer = str(x.get('response', x.get('answer', x.get('Answer', ''))))
            assert text and answer, x
            rs.append(record('mathoctopus', lang, 'test', 'msvamp:' + str(x.get('ID', x.get('id', i))), text, answer, gold_numeric=number(answer), benchmark='msvamp'))
        p = SRC / f'google-research-datasets--MGSM-Rev2/mgsm_{lang}.tsv'
        for i, row in enumerate(csv.reader(p.open(), delimiter='\t')):
            assert len(row) >= 2, row
            assert number(row[1]) is not None, row
            rs.append(record('mathoctopus', lang, 'test', f'mgsm_rev2:{i}', row[0], row[1], gold_numeric=number(row[1]), benchmark='mgsm_rev2'))
    return (rs, dict(source_sha256=sha(path), source_counts=dict(counts), rejected_blocks=dict(rejected), valid_blocks=len(selected), alignment='RECONSTRUCTED English-led blocks, exact official GSM8K-train question identity and numeric agreement in all four languages. Not released semantic IDs; numeric agreement does not prove translation semantics. Same-ID interpretation qualified.', license='Apache-2.0 (MathOctopus); MGSM-Rev2 upstream terms must be retained.', internal_dev=256))

def xlsum():
    rs = []
    counts = {}
    for lang, name in zip(LANGS, ['spanish', 'thai', 'french', 'english']):
        p = SRC / f'csebuetnlp--xlsum/data/{name}_XLSum_v2.0.tar.bz2'
        with tarfile.open(p) as tf:
            for mem in tf.getmembers():
                if not mem.isfile() or not mem.name.endswith('.jsonl'):
                    continue
                split = 'dev' if 'val' in mem.name else 'test' if 'test' in mem.name else 'train'
                n = 0
                for line in tf.extractfile(mem):
                    x = json.loads(line)
                    rs.append(record('xlsum', lang, split, lang + ':' + x['id'], x['text'], x['summary']))
                    n += 1
                counts[lang + ':' + split] = n
    return (rs, dict(alignment='NONPARALLEL official language-specific BBC articles; no Same-ID claim.', license='CC-BY-NC-SA-4.0', source_counts=counts))

def mtop():
    rs = []
    p = SRC / 'facebook--mtop/mtop.zip'
    with zipfile.ZipFile(p) as z:
        for lang in LANGS:
            for split, name in [('train', 'train'), ('dev', 'eval'), ('test', 'test')]:
                for line in z.read(f'mtop/{lang}/{name}.txt').decode().splitlines():
                    fields = line.split('\t')
                    assert len(fields) == 8
                    id, intent, slot, text, domain, locale, tree, tokens = fields
                    assert tree.count('[') == tree.count(']') and tree.startswith('[' + intent + ' ')
                    rs.append(record('mtop', lang, split, id, text, tree, root_intent=intent, domain=domain))
    return (rs, dict(source_sha256=sha(p), alignment='Official v2 IDs explicitly identify translations (release README). Native compositional decoupled column7, whitespace canonicalized by parser; approximate parallelism, not perfect translation semantics.', license='CC-BY-SA-4.0', serialization='Released column7; no flattening, no artificial tree conversion.'))
LOADERS = {'massive': massive, 'mathoctopus': mathoctopus, 'xnli': xnli, 'mtop': mtop, 'xlsum': xlsum, 'sib200': sib200}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--track', choices=list(LOADERS), required=True)
    p.add_argument('--data-version', default='prepared-v1')
    a = p.parse_args()
    out = DATA / a.data_version / a.track
    assert not out.exists(), out
    rs, meta = LOADERS[a.track]()
    raw_counts = collections.Counter(((r['lang'], r['split']) for r in rs))
    seen = {}
    rejected = collections.Counter()
    for r in rs:
        if not norm(r['text']) or not str(r['answer']).strip():
            rejected['empty'] += 1
            continue
        key = (r['lang'], r['split'], r['id'])
        if key in seen:
            assert seen[key]['answer'] == r['answer'], ('Conflicting duplicate ID', key)
            rejected['duplicate_id'] += 1
            continue
        seen[key] = r
    rs = list(seen.values())
    label_tracks = ['massive', 'xnli', 'sib200']
    labels = sorted({r['answer'] for r in rs if r['split'] == 'train'}) if a.track in label_tracks else []
    if a.track == 'mtop':
        labels = sorted({label for r in rs if r['split'] == 'train' for label in re.findall('(?:IN|SL):[A-Z_]+', r['answer'])})
    maps = {lang: {sp: {r['id']: r for r in rs if r['lang'] == lang and r['split'] == sp} for sp in ['train', 'dev', 'test']} for lang in LANGS}
    aligned = a.track != 'xlsum'
    cross_conflicts = []
    if aligned:
        for sp in ['train', 'dev', 'test']:
            common = set.intersection(*[set(maps[l][sp]) for l in LANGS])
            if a.track in label_tracks:
                cross_conflicts.extend(((sp, id) for id in common if len({maps[l][sp][id]['answer'] for l in LANGS}) > 1))
    assert not cross_conflicts, cross_conflicts[:5]
    blocked = set()
    overlap_counts = {}
    for lang in LANGS:
        eval_texts = {norm(r['text']) for r in rs if r['lang'] == lang and r['split'] in ['dev', 'test']}
        local = {r['id'] for r in maps[lang]['train'].values() if norm(r['text']) in eval_texts}
        overlap_counts[lang] = len(local)
        blocked |= local
    train_common = set.intersection(*[set(maps[l]['train']) for l in LANGS]) if aligned else set()
    ids = order(train_common - blocked, a.track + '-train') if aligned else None
    n = 640 if a.track == 'sib200' else 3200
    if aligned:
        n = min(n, len(ids) // 16 * 16)
    assert n >= 16
    subset_ids = {l: ids[:n] if aligned else order(set(maps[l]['train']) - blocked, a.track + l + '-train')[:n] for l in LANGS}
    dev_common = set.intersection(*[set(maps[l]['dev']) for l in LANGS]) if aligned else set()
    dev_ids = order(dev_common, a.track + '-dev')[:256] if aligned else None
    out.mkdir(parents=True)
    from transformers import AutoTokenizer
    from llamafactory.hparams import DataArguments
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from datasets import Dataset, DatasetDict
    import numpy as np
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    args = DataArguments(template='qwen3', enable_thinking=False, train_on_prompt=False, cutoff_len=2048, packing=False)
    template = get_template_and_fix_tokenizer(tok, args)
    processor = SupervisedDatasetProcessor(template, tok, None, args)
    files = {}
    token_stats = {}
    dataset_info = {}
    for lang in LANGS:
        for sp in ['train', 'dev', 'test']:
            if sp == 'train':
                chosen = [maps[lang][sp][i] for i in subset_ids[lang]]
            elif sp == 'dev':
                chosen = [maps[lang][sp][i] for i in dev_ids] if aligned else [maps[lang][sp][i] for i in order(maps[lang][sp], a.track + lang + '-dev')[:256]]
            else:
                chosen = list(maps[lang][sp].values())
            features = []
            counts = []
            ready = []
            for r in chosen:
                r = dict(r)
                r['prompt'] = prompt(a.track, lang, r['text'], labels)
                r['labels'] = labels
                msg = [{'role': 'user', 'content': r['prompt']}, {'role': 'assistant', 'content': r['answer']}]
                source, target = template.encode_multiturn(tok, msg, None, None)[0]
                ids2, labs = processor._encode_data_example(msg[:1], msg[1:], None, None, [], [], [])
                assert any((t != -100 for t in labs[1:])) and any((t == -100 for t in labs))
                r['prompt_token_ids'] = source
                r['input_tokens_untruncated'] = len(source) + len(target)
                r['assistant_tokens_untruncated'] = len(target)
                r['training_input_tokens'] = len(ids2)
                r['training_supervised_tokens'] = sum((t != -100 for t in labs[1:]))
                r['source_truncated'] = len(ids2) < len(source) + len(target)
                assert len(source) <= 6144, ('Long inference source; audit before protocol', a.track, lang, sp, len(source))
                if sp == 'train':
                    features.append({'input_ids': ids2, 'attention_mask': [1] * len(ids2), 'labels': labs})
                counts.append([len(source) + len(target), len(target), len(ids2), sum((t != -100 for t in labs[1:]))])
                ready.append(r)
            assert len(ready) == len(chosen)
            f = out / f'{lang}_{sp}.jsonl'
            write_rows(f, ready)
            files[f.name] = sha(f)
            if sp == 'train':
                DatasetDict({'train': Dataset.from_list(features)}).save_to_disk(str(out / f'{lang}_tokenized'))
                name = lang + '_sft'
                write_rows(out / (name + '.jsonl'), [{'messages': [{'role': 'user', 'content': r['prompt']}, {'role': 'assistant', 'content': r['answer']}]} for r in ready])
                dataset_info[name] = {'file_name': name + '.jsonl', 'formatting': 'sharegpt', 'columns': {'messages': 'messages'}, 'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'}}
            if counts:
                ar = np.asarray(counts)
                token_stats[lang + ':' + sp] = {'n': len(counts), 'totals': ar.sum(0).tolist(), 'quantiles_0_50_90_95_99_100': np.percentile(ar, [0, 50, 90, 95, 99, 100], axis=0).tolist(), 'truncated': sum((r['source_truncated'] for r in ready))}
            del ready
    dump(out / 'dataset_info.json', dataset_info)
    protocol = dict(track=a.track, status='P0_PREPARED_PENDING_P1', model_revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e', tokenizer_sha256=sha(MODEL / 'tokenizer.json'), metadata=meta, raw_counts={l + ':' + s: n for (l, s), n in raw_counts.items()}, rejected=dict(rejected), exact_train_eval_overlap_excluded=overlap_counts, common_training_ids=len(train_common), n_train_per_language=n, steps=n // 16, dev_per_language=len(dev_ids) if aligned else 256, dev_scope='Frozen SHA-seeded subset of official dev; math separate train-source256 IDs', same_id=aligned, training_ids=subset_ids['es'] if aligned else subset_ids, disjoint_reserve_ids=ids[n:2 * n] if aligned and len(ids) >= 2 * n else [], token_stats_columns=['input_untruncated', 'assistant_untruncated', 'input_after_LF_cutoff', 'shifted_supervised_after_LF_cutoff'], token_stats=token_stats, files=files, labels=labels, split_policy='Official dev/test; train exact normalized text overlaps excluded jointly across languages; no near-duplicate search. Training rows SHA-seeded; final loader seed42 frozen.', test_exposure='New track formal test NOT_RUN; possible historical source/benchmark exposure disclosed separately; no pretraining-contamination guarantee.')
    dump(out / 'protocol.json', protocol)
    print(json.dumps({k: protocol[k] for k in ['track', 'status', 'raw_counts', 'common_training_ids', 'n_train_per_language', 'steps']}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
