"""XL-Sum native splits with a predeclared 1536-source/512-target token budget.

Preserve the chat header when clipping articles; do not clip formal references.
The 2048 training cutoff is unchanged. Source-prefix information loss is reported.
"""
from runtime import runtime_path
import collections, json, os, tarfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
from common import ROOT, DATA, MODEL, LANGS, sha, oh, dump, norm, order, write_rows, prompt

def main():
    from transformers import AutoTokenizer
    from datasets import Dataset, DatasetDict
    from llamafactory.hparams import DataArguments
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    out = DATA / 'prepared-v1/xlsum'
    assert not out.exists()
    out.mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    args = DataArguments(template='qwen3', enable_thinking=False, train_on_prompt=False, cutoff_len=2048, packing=False)
    template = get_template_and_fix_tokenizer(tok, args)
    processor = SupervisedDatasetProcessor(template, tok, None, args)
    audit = Path(runtime_path('results', '')) / 'xlsum/raw_audit_v1.json'
    assert json.loads(audit.read_text())['status'] == 'AUDIT_DONE_PENDING_TRUNCATION_PROTOCOL'
    files = {}
    stats = {}
    train_ids = {}
    info = {}
    source_files = {}
    truncation = {}
    counts = {}
    for lang, name in zip(LANGS, ['spanish', 'thai', 'french', 'english']):
        p = DATA / f'sources/csebuetnlp--xlsum/data/{name}_XLSum_v2.0.tar.bz2'
        source_files[p.name] = sha(p)
        groups = {}
        with tarfile.open(p) as tf:
            for mem in tf.getmembers():
                if not mem.isfile() or not mem.name.endswith('.jsonl'):
                    continue
                split = 'dev' if 'val' in mem.name else 'test' if 'test' in mem.name else 'train'
                groups[split] = [json.loads(line) for line in tf.extractfile(mem)]
        eval_norms = {oh(norm(r['text'])) for s in ['dev', 'test'] for r in groups[s]}
        chosen = {}
        eligible = {str(r['id']): r for r in groups['train'] if r['text'].strip() and r['summary'].strip() and (oh(norm(r['text'])) not in eval_norms)}
        ids = order(eligible, 'xlsum' + lang + '-train')[:3200]
        assert len(ids) == 3200
        chosen['train'] = [eligible[i] for i in ids]
        train_ids[lang] = ids
        dev = {str(r['id']): r for r in groups['dev']}
        chosen['dev'] = [dev[i] for i in order(dev, 'xlsum' + lang + '-dev')[:256]]
        chosen['test'] = groups['test']
        counts[lang] = {s: len(rs) for s, rs in groups.items()}
        for split, raws in chosen.items():
            ready = []
            features = []
            vals = []
            coverage = []
            for raw in raws:
                text = raw['text']
                target = raw['summary']
                full_prompt = prompt('xlsum', lang, text)
                msg = [{'role': 'user', 'content': full_prompt}, {'role': 'assistant', 'content': target}]
                full_src, full_tgt = template.encode_multiturn(tok, msg, None, None)[0]
                src = full_src
                article_truncated = False
                if len(src) > 1536:
                    article_ids = tok.encode(text, add_special_tokens=False)
                    budget = max(1, len(article_ids) - (len(src) - 1536))
                    while True:
                        clipped = tok.decode(article_ids[:budget], skip_special_tokens=False)
                        current = prompt('xlsum', lang, clipped)
                        msg = [{'role': 'user', 'content': current}, {'role': 'assistant', 'content': target}]
                        src, tgt = template.encode_multiturn(tok, msg, None, None)[0]
                        if len(src) <= 1536:
                            break
                        budget -= max(1, len(src) - 1536)
                    article_truncated = True
                else:
                    current = full_prompt
                train_target = target
                if len(full_tgt) > 512:
                    target_ids = tok.encode(target, add_special_tokens=False)
                    target_budget = 480
                    while True:
                        train_target = tok.decode(target_ids[:target_budget], skip_special_tokens=False)
                        msg[1]['content'] = train_target
                        _, bounded_target = template.encode_multiturn(tok, msg, None, None)[0]
                        if len(bounded_target) <= 512:
                            break
                        target_budget -= max(1, len(bounded_target) - 512)
                ids2, labs = processor._encode_data_example(msg[:1], msg[1:], None, None, [], [], [])
                assert len(src) <= 1536 and 0 < sum((x != -100 for x in labs[1:])) <= 512
                r = dict(track='xlsum', lang=lang, split=split, id=lang + ':' + str(raw['id']), prompt=current, answer=target, training_answer=train_target, prompt_token_ids=src, input_tokens_untruncated=len(full_src) + len(full_tgt), assistant_tokens_untruncated=len(full_tgt), training_input_tokens=len(ids2), training_supervised_tokens=sum((x != -100 for x in labs[1:])), source_truncated=article_truncated, source_tokens_untruncated=len(full_src), source_tokens_used=len(src), source_coverage_ratio=len(src) / len(full_src), target_truncated=len(labs) - len(src) < len(full_tgt), labels=[])
                ready.append(r)
                vals.append([len(full_src) + len(full_tgt), len(full_tgt), len(ids2), r['training_supervised_tokens']])
                coverage.append(r['source_coverage_ratio'])
                if split == 'train':
                    features.append({'input_ids': ids2, 'attention_mask': [1] * len(ids2), 'labels': labs})
            f = out / f'{lang}_{split}.jsonl'
            write_rows(f, ready)
            files[f.name] = sha(f)
            ar = np.array(vals)
            stats[lang + ':' + split] = {'n': len(ready), 'totals': ar.sum(0).tolist(), 'quantiles_0_50_90_95_99_100': np.percentile(ar, [0, 50, 90, 95, 99, 100], axis=0).tolist(), 'truncated': sum((r['source_truncated'] for r in ready)), 'target_truncated': sum((r['target_truncated'] for r in ready))}
            truncation[lang + ':' + split] = {'source_cap_including_header': 1536, 'target_cap_training_only': 512, 'source_truncated_n': stats[lang + ':' + split]['truncated'], 'coverage_quantiles': np.percentile(coverage, [0, 25, 50, 75, 100]).tolist()}
            if split == 'train':
                DatasetDict({'train': Dataset.from_list(features)}).save_to_disk(str(out / f'{lang}_tokenized'))
                name = lang + '_sft'
                write_rows(out / (name + '.jsonl'), [{'messages': [{'role': 'user', 'content': r['prompt']}, {'role': 'assistant', 'content': r['training_answer']}]} for r in ready])
                info[name] = {'file_name': name + '.jsonl', 'formatting': 'sharegpt', 'columns': {'messages': 'messages'}, 'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'}}
            print(json.dumps({'lang': lang, 'split': split, 'n': len(ready), 'source_truncated': stats[lang + ':' + split]['truncated']}), flush=True)
        del groups, chosen, eligible
    dump(out / 'dataset_info.json', info)
    protocol = dict(track='xlsum', status='P0_PREPARED_PENDING_P1', model_revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e', tokenizer_sha256=sha(MODEL / 'tokenizer.json'), metadata=dict(alignment='NONPARALLEL native BBC articles. No cross-language ID alignment or pure language causality.', license='CC-BY-NC-SA-4.0', source_files=source_files, source_revision='30fece425f9a3866e04321773ca7a80056d55ca6'), raw_counts={l + ':' + s: n for l, ss in counts.items() for s, n in ss.items()}, common_training_ids=0, n_train_per_language=3200, steps=200, dev_per_language=256, same_id=False, training_ids=train_ids, disjoint_reserve_ids=[], token_stats=stats, truncation=truncation, files=files, labels=[], source_truncation_policy='Before chat rendering, keep the longest article TOKEN PREFIX that fits 1536 source tokens including instruction and intact assistant header; same across training/dev/test and all models. Training total2048, targets at most512 under upstream LF cutoff. Full formal references retained. Generated max512 greedy.', scoring='Author multilingual ROUGE afcd803c8d7c98f3be60e9a7ce57fcbc3e729e8c; Spanish/French/English Snowball stemming, Thai bundled PyThaiNLP newmm; rouge1/2/Lsum F1, with newline boundaries as provided.', test_exposure='New language task protocol frozen before any model scoring. Earlier project XL-Sum SW/EN exposure exists; do not claim fully unexposed English source test.')
    dump(out / 'protocol.json', protocol)
if __name__ == '__main__':
    main()
