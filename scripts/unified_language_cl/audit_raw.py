"""Seven-track source audit snapshot, independent of model scores.

Complete source integrity and exact split overlap, without renewed full-corpus
near-duplicate or model-based language audits. Reports unavailable author split
files as blocked rather than inventing a new formal test.
"""
from runtime import runtime_path
import argparse, collections, json, os, re, tarfile, time, zipfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from common import ROOT, DATA, MODEL, LANGS, sha, oh, dump, norm, order
import pyarrow.parquet as pq
SRC = DATA / 'sources'

def wikilingua():
    counts = {}
    ids = {}
    duplicates = {}
    nonempty = {}
    examples = {}
    for lang, name in zip(LANGS[:3], ['spanish', 'thai', 'french']):
        p = SRC / f'esdurmus--wiki_lingua/{name}/train-00000-of-00001.parquet'
        if not p.exists():
            return {'status': 'DOWNLOADING', 'reason': 'raw language file incomplete'}
        seen = set()
        total = 0
        empty = 0
        dups = 0
        els = []
        for batch in pq.ParquetFile(p).iter_batches(batch_size=512):
            for r in batch.to_pylist():
                a = r['article']
                for i in range(len(a['document'])):
                    total += 1
                    sid = oh([a['english_url'][i], a['english_section_name'][i]])
                    if sid in seen:
                        dups += 1
                    seen.add(sid)
                    if not a['document'][i].strip() or not a['summary'][i].strip():
                        empty += 1
                    els.append((sid, a['document'][i], a['summary'][i]))
        counts[lang] = {'raw_article_records': pq.ParquetFile(p).metadata.num_rows, 'raw_section_pairs': total, 'official_dev': None, 'official_test': None, 'file_sha256': sha(p)}
        ids[lang] = seen
        duplicates[lang] = dups
        nonempty[lang] = total - empty
        examples[lang] = sorted(els, key=lambda x: oh([42, x[0]]))[:256]
    from transformers import AutoTokenizer
    import numpy as np
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    lengths = {}
    for l, items in examples.items():
        vals = [[len(tok.encode(x, add_special_tokens=False)), len(tok.encode(y, add_special_tokens=False))] for _, x, y in items]
        arr = np.array(vals)
        lengths[l] = {'sample_n': len(vals), 'columns': ['article_tokens', 'summary_tokens'], 'quantiles_0_50_90_95_99_100': np.percentile(arr, [0, 50, 90, 95, 99, 100], axis=0).tolist(), 'raw_pair_over2048': int((arr.sum(1) > 2048).sum())}
    return {'status': 'BLOCKED_OFFICIAL_SPLITS', 'counts': counts, 'common_es_th_fr_section_ids': len(set.intersection(*ids.values())), 'duplicate_section_id_records': duplicates, 'nonempty_pairs': nonempty, 'raw_token_sample': lengths, 'alignment': 'English URL + aligned English section name in publisher raw fields. Image alignments not independently re-created from inaccessible original archive.', 'reason': 'HF collection has only a single train split. Author explicitly says updated leakage-safe train/test archive is at Google Drive 1sTCB5NDPq6vUOlxR29DbvSssErvXLD1d; direct request returned HTTP404. No invented official dev/test or unreviewed third-party substitute.', 'license': 'HF card CC-BY-3.0 differs from author README wikiHow attribution CC-BY-NC-SA-3.0; retain stricter original content terms, no raw-text redistribution.'}

def xlsum():
    counts = {}
    ids = {}
    overlap = {}
    duplicates = {}
    sample = {}
    for lang, name in zip(LANGS, ['spanish', 'thai', 'french', 'english']):
        p = SRC / f'csebuetnlp--xlsum/data/{name}_XLSum_v2.0.tar.bz2'
        if not p.exists():
            return {'status': 'DOWNLOADING', 'reason': str(p.name) + ' incomplete', 'completed_language_counts': counts}
        local = {}
        texts = {}
        els = []
        dups = 0
        with tarfile.open(p) as tf:
            for m in tf.getmembers():
                if not m.isfile() or not m.name.endswith('.jsonl'):
                    continue
                split = 'dev' if 'val' in m.name else 'test' if 'test' in m.name else 'train'
                seen = set()
                seen_text = set()
                n = 0
                empty = 0
                for line in tf.extractfile(m):
                    r = json.loads(line)
                    n += 1
                    id = r['id']
                    dups += id in seen
                    seen.add(id)
                    seen_text.add(oh(norm(r['text'])))
                    if not r['text'].strip() or not r['summary'].strip():
                        empty += 1
                    if split == 'train':
                        els.append((id, r['text'], r['summary']))
                local[split] = seen
                texts[split] = seen_text
                counts[lang + ':' + split] = {'n': n, 'empty': empty}
        overlap[lang] = {f'{a}:{b}': {'id': len(local[a] & local[b]), 'exact_article_text': len(texts[a] & texts[b])} for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]}
        ids[lang] = local
        duplicates[lang] = dups
        sample[lang] = sorted(els, key=lambda x: oh([42, x[0]]))[:256]
    from transformers import AutoTokenizer
    import numpy as np
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    lengths = {}
    for l, items in sample.items():
        ar = np.array([[len(tok.encode(x, add_special_tokens=False)), len(tok.encode(y, add_special_tokens=False))] for _, x, y in items])
        lengths[l] = {'n': len(items), 'columns': ['article_tokens', 'summary_tokens'], 'quantiles_0_50_90_95_99_100': np.percentile(ar, [0, 50, 90, 95, 99, 100], axis=0).tolist(), 'raw_pair_over2048': int((ar.sum(1) > 2048).sum())}
    return dict(status='AUDIT_DONE_PENDING_TRUNCATION_PROTOCOL', counts=counts, exact_overlaps=overlap, duplicates=duplicates, alignment='Nonparallel language-native BBC articles. No Same-ID experiment or language-only causal inference.', token_sample=lengths, license='CC-BY-NC-SA-4.0', summary_scoring_note='Default rouge-score regex discards Thai script; use an explicit Unicode/Thai-aware tokenizer and disclose metric comparability before any score.')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', choices=['wikilingua', 'xlsum'], required=True)
    a = ap.parse_args()
    value = globals()[a.track]()
    dump(Path(runtime_path('results', '')) / f'{a.track}/raw_audit_v1.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
