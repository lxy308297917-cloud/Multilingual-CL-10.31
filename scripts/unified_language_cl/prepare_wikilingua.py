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
    out = DATA / 'prepared-v1/wikilingua'
    assert not out.exists()
    out.mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    args = DataArguments(template='qwen3', enable_thinking=False, train_on_prompt=False, cutoff_len=2048, packing=False)
    template = get_template_and_fix_tokenizer(tok, args)
    processor = SupervisedDatasetProcessor(template, tok, None, args)
    audit = Path(runtime_path('results', '')) / 'wikilingua/gem_raw_audit_v1.json'
    assert json.loads(audit.read_text())['status'] == 'GEM_SPLITS_AUDITED_PENDING_PREPARATION'
    rawdir = DATA / 'gem-wikilingua-audit-v2'
    all_rows = {sp: [json.loads(line) for line in (rawdir / (sp + '.jsonl')).open()] for sp in ['train', 'dev', 'test']}
    held_articles = {r['english_article_url'] for sp in ['dev', 'test'] for r in all_rows[sp]}
    eligible_by_lang = {l: {r['semantic_id']: r for r in all_rows['train'] if r['lang'] == l and r['english_article_url'] not in held_articles} for l in LANGS}
    common = set.intersection(*(set(m) for m in eligible_by_lang.values()))
    shared_ids = order(common, 'wikilingua-gem-common-train')[:3200]
    assert len(shared_ids) == 3200
    dev_by_lang = {l: {r['semantic_id']: r for r in all_rows['dev'] if r['lang'] == l} for l in LANGS}
    dev_common = set.intersection(*(set(m) for m in dev_by_lang.values()))
    dev_ids = order(dev_common, 'wikilingua-gem-common-dev')[:256]
    assert len(dev_ids) == 256
    files = {}
    stats = {}
    train_ids = {}
    info = {}
    source_files = {'wikilingua_cleaned.tar.gz': '95663019cebc422c9b017f6b670a67a272eef259ce69d63677bf4606b692ed83'}
    truncation = {}
    counts = {}
    for lang in LANGS:

        def native(r):
            return dict(id=r['semantic_id'], text=r['text'], summary=r['answer'])
        chosen = {'train': [native(eligible_by_lang[lang][i]) for i in shared_ids], 'dev': [native(dev_by_lang[lang][i]) for i in dev_ids], 'test': [native(r) for r in all_rows['test'] if r['lang'] == lang]}
        train_ids[lang] = shared_ids
        counts[lang] = {sp: sum((r['lang'] == lang for r in all_rows[sp])) for sp in all_rows}
        for split, raws in chosen.items():
            ready = []
            features = []
            vals = []
            coverage = []
            for raw in raws:
                text = raw['text']
                target = raw['summary']
                full_prompt = prompt('wikilingua', lang, text)
                msg = [{'role': 'user', 'content': full_prompt}, {'role': 'assistant', 'content': target}]
                full_src, full_tgt = template.encode_multiturn(tok, msg, None, None)[0]
                src = full_src
                article_truncated = False
                if len(src) > 1536:
                    article_ids = tok.encode(text, add_special_tokens=False)
                    budget = max(1, len(article_ids) - (len(src) - 1536))
                    while True:
                        clipped = tok.decode(article_ids[:budget], skip_special_tokens=False)
                        current = prompt('wikilingua', lang, clipped)
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
                r = dict(track='wikilingua', lang=lang, split=split, id=lang + ':' + str(raw['id']), prompt=current, answer=target, training_answer=train_target, prompt_token_ids=src, input_tokens_untruncated=len(full_src) + len(full_tgt), assistant_tokens_untruncated=len(full_tgt), training_input_tokens=len(ids2), training_supervised_tokens=sum((x != -100 for x in labs[1:])), source_truncated=article_truncated, source_tokens_untruncated=len(full_src), source_tokens_used=len(src), source_coverage_ratio=len(src) / len(full_src), target_truncated=len(labs) - len(src) < len(full_tgt), labels=[])
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
        del chosen
    dump(out / 'dataset_info.json', info)
    protocol = dict(track='wikilingua', status='P0_PREPARED_PENDING_P1', model_revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e', tokenizer_sha256=sha(MODEL / 'tokenizer.json'), metadata=dict(alignment='Exact normalized source+summary matched to original publisher English URL+section; shared IDs for ES/TH/FR/EN training and dev. Never aligned by row position.', license='CC-BY-NC-SA-3.0', source_files=source_files, source_revision='af5d0f00b59a6933165c97b384f50d8b563c314d', official_source='Author-supported GEM WikiLingua multilingual official splits; uniquely mapped nonempty subset, NOT full official test.'), raw_counts={l + ':' + sp: n for l, ss in counts.items() for sp, n in ss.items()}, common_training_ids=len(common), n_train_per_language=3200, steps=200, dev_per_language=256, same_id=True, training_ids=train_ids, disjoint_reserve_ids=[], token_stats=stats, truncation=truncation, files=files, labels=[], source_truncation_policy='Same as frozen XL-Sum: source token prefix <=1536 including intact header; training target <=512; total2048. Full formal reference retained; greedy generation512.', scoring='Author XL-Sum multilingual ROUGE afcd803c8d7c98f3be60e9a7ce57fcbc3e729e8c, ES/FR/EN Snowball, TH newmm. Not exact GEM leaderboard scoring.', split_audit_sha256=sha(audit), limitations=['All matched English article URLs in dev/test excluded jointly from training.', 'GEM dev/test share some English articles through different sections; official section splits retained. Not article-independent dev/test.', 'Unmatched and ambiguous original mappings excluded BEFORE inference; report exact counts and coverage. Formal result is mapped subset, not full official test.', 'Only 3985 common eligible IDs, so no 6400-example disjoint experiment.'], test_exposure='No unified WikiLingua model scoring before protocol freeze.')
    dump(out / 'protocol.json', protocol)
if __name__ == '__main__':
    main()
