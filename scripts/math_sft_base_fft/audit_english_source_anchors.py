"""Bounded exact source-identity check, no scores, no mutation of frozen assets."""
from profile_paths import resolve
import sys, json, csv, urllib.request, concurrent.futures
from pathlib import Path
from common import *
import pyarrow.parquet as pq
aux = DATA / 'audit-english-anchors-v1'
assert not aux.exists()
aux.mkdir()
poly_rev = json.loads((DATA / 'sources.lock.json').read_text())['polymath_dataset_revision']

def fetch(level):
    p = aux / (level + '.parquet')
    u = f'https://huggingface.co/datasets/Qwen/PolyMath/resolve/{poly_rev}/en/{level}.parquet'
    with urllib.request.urlopen(u, timeout=90) as r:
        b = r.read()
    p.write_bytes(b)
    return {'level': level, 'sha256': sha(p), 'url': u, 'file': p}
assets = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    assets = list(ex.map(fetch, ['low', 'medium', 'high', 'top']))
anchors = {}
counts = {}

def add(task, qs):
    counts[task] = len(qs)
    for q in qs:
        anchors.setdefault(oh(norm(q)), []).append(task)
cache = Path(resolve('LEGACY_DATA') + '/sources')
canon = Path(resolve('BENCH_CACHE') + '/gsm8k/740312add88f781978c0658806c59bc2815b9866/main')
add('en_gsm8k_test', [x['question'] for x in pq.read_table(canon / 'test-00000-of-00001.parquet').to_pylist()])
add('en_mgsm_rev2', [x[0] for x in csv.reader((cache / 'google-research-datasets--MGSM-Rev2/mgsm_en.tsv').open(), delimiter='\t')])
qs = []
for x in rows(cache / 'Mathoctopus--MSVAMP/test_English.json'):
    q = x.get('m_query', x.get('question', x.get('Question', x.get('Body', ''))))
    if 'Body' in x and 'Question' in x:
        q = x['Body'] + ' ' + x['Question']
    assert q
    qs.append(q)
add('en_msvamp', qs)
for x in assets:
    add('en_polymath_' + x['level'], [v['question'] for v in pq.read_table(x['file']).to_pylist()])
hits = {}
for lang in LANGS:
    hs = [{'source_id': r['id'], 'benchmark_sources': anchors[r['source_question_sha256']]} for r in rows(DATA / 'mathoctopus-audit-v1' / f'{lang}_all_valid.jsonl') if r['source_question_sha256'] in anchors]
    hits[lang] = hs
report = {'status': 'COMPLETE_EXACT_SOURCE_ANCHOR_CHECK', 'frozen_training_data_unchanged': True, 'anchor_counts': counts, 'assets': [{'level': x['level'], 'sha256': x['sha256'], 'url': x['url']} for x in assets], 'hits_by_language': hits, 'rule': 'exact NFKC-casefold-whitespace canonical English source question identity; not numerical-answer matching', 'scope': 'Auxiliary English source anchors complement the existing nine-language exact question check; no English PolyMath performance evaluated; no semantic/paraphrase contamination-free claim'}
dump(aux / 'report.json', report)
dump(ROOT / 'results/math_sft_base_fft/english_source_anchor_audit.json', report)
print(json.dumps({'anchor_counts': counts, 'hits': {k: len(v) for k, v in hits.items()}}, indent=2))
