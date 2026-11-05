"""Hash-verified, bounded-concurrency mCoT ingestion for source-group auditing.

Downloads only the 20 JSONL shards; never the duplicate merged JSON array.
Retains exact EN/SW/BN/JA records in SQLite, with per-shard provenance. Removes
only this tool's verified temporary shard after its indexed transaction commits.
"""
import concurrent.futures as cf, collections, hashlib, json, os, re, shutil, sqlite3, time, urllib.request
from pathlib import Path
from common import *
REV = '3fbab9c3f70784fb5d9328233375c9d8cfadf1f7'
KEEP = {'en', 'sw', 'bn', 'ja'}

def main():
    gate = ROOT / 'results/math_sft_base_fft/routeA-math-gate-v1/decision.json'
    assert json.loads(gate.read_text())['status'] == 'FAIL_FIXED_E1_AND_E2_GATE'
    cfg = json.loads((ROOT / 'configs/math_sft_base_fft/mcot_shards.lock.json').read_text())
    parts = cfg['parts']
    assert len(parts) == 20 and cfg['revision'] == REV
    out = DATA / 'mcot-source-audit-v1'
    out.mkdir(exist_ok=True)
    temp = out / 'verified-temporary-shards'
    temp.mkdir(exist_ok=True)
    log = out / 'download-progress.jsonl'
    conn = sqlite3.connect(out / 'raw_selected_languages.sqlite3')
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA cache_size=-131072')
    conn.execute('CREATE TABLE IF NOT EXISTS raw (lang TEXT NOT NULL,record_id TEXT NOT NULL,question TEXT NOT NULL,answer TEXT NOT NULL,qhash TEXT NOT NULL,ahash TEXT NOT NULL,source_part TEXT NOT NULL,PRIMARY KEY(lang,record_id))')
    conn.execute('CREATE TABLE IF NOT EXISTS parts (name TEXT PRIMARY KEY,sha256 TEXT NOT NULL,size INTEGER NOT NULL,stats TEXT NOT NULL)')
    conn.commit()
    done = {r[0]: r[1] for r in conn.execute('SELECT name,sha256 FROM parts')}
    todo = [p for p in parts if p['path'] not in done]
    for p in parts:
        if p['path'] in done:
            assert done[p['path']] == p['sha256']

    def progress(x):
        with log.open('a') as f:
            f.write(json.dumps(x) + '\n')

    def room():
        assert shutil.disk_usage(out).free > 41 * 1024 ** 3, 'Preserve >=31GiB future checkpoint capacity and >=10GiB storage margin; no silent data-budget cut'

    def fetch(p):
        room()
        path = temp / p['path']
        if path.exists():
            assert path.stat().st_size == p['size'] and sha(path) == p['sha256']
            return (p, path)
        partial = path.with_suffix(path.suffix + '.partial')
        assert not partial.exists(), f'Inspect/preserve incomplete download before recovery: {partial}'
        url = f"https://huggingface.co/datasets/laihuiyuan/mCoT-MATH/resolve/{REV}/{p['path']}?download=true"
        h = hashlib.sha256()
        count = 0
        start = time.monotonic()
        req = urllib.request.Request(url, headers={'User-Agent': 'Multilingual-CL-pinned-data-ingestion/1.0'})
        with urllib.request.urlopen(req, timeout=90) as response, partial.open('xb') as f:
            while True:
                block = response.read(1024 ** 2)
                if not block:
                    break
                f.write(block)
                h.update(block)
                count += len(block)
                if count % (32 * 1024 ** 2) == 0:
                    progress({'part': p['path'], 'bytes': count, 'total': p['size'], 'seconds': time.monotonic() - start})
        assert count == p['size'] and h.hexdigest() == p['sha256'], (p['path'], count, h.hexdigest())
        partial.rename(path)
        progress({'part': p['path'], 'status': 'DOWNLOADED_VERIFIED', 'bytes': count, 'seconds': time.monotonic() - start})
        return (p, path)

    def ingest(p, path):
        room()
        h = hashlib.sha256()
        counts = collections.Counter()
        stats = collections.Counter()
        buffer = []
        conn.execute('BEGIN')
        try:

            def insert():
                for lang, rid, q, a, qh, ah, part in buffer:
                    cur = conn.execute('INSERT OR IGNORE INTO raw VALUES (?,?,?,?,?,?,?)', (lang, rid, q, a, qh, ah, part))
                    if cur.rowcount == 0:
                        prev = conn.execute('SELECT qhash,ahash FROM raw WHERE lang=? AND record_id=?', (lang, rid)).fetchone()
                        assert prev == (qh, ah), ('conflicting duplicate language/counter identity', lang, rid)
                        stats['duplicate_identical_records'] += 1
                buffer.clear()
            with path.open('rb') as f:
                for raw in f:
                    h.update(raw)
                    if not raw.strip():
                        stats['empty_lines'] += 1
                        continue
                    stats['nonempty_lines'] += 1
                    try:
                        x = json.loads(raw)
                    except (ValueError, UnicodeDecodeError):
                        stats['invalid_json_lines'] += 1
                        continue
                    ident = x.get('lang-id')
                    m = re.fullmatch('([a-z]{2})-(\\d+)', ident or '')
                    if not m:
                        stats['unrecoverable_lang_id'] += 1
                        continue
                    lang, rid = m.groups()
                    counts[lang] += 1
                    if lang not in KEEP:
                        continue
                    q = x.get('question')
                    a = x.get('answer')
                    if not isinstance(q, str) or not isinstance(a, str) or (not q.strip()) or (not a.strip()):
                        stats['damaged_empty_question_or_answer'] += 1
                        continue
                    buffer.append((lang, rid, q, a, oh(norm(q)), oh(norm(a)), p['path']))
                    stats['selected_language_records'] += 1
                    if len(buffer) >= 2000:
                        insert()
            insert()
            assert h.hexdigest() == p['sha256']
            detail = {'record_counts_by_language': dict(counts), 'counts': dict(stats), 'indexed_actual_file_sha256': h.hexdigest()}
            conn.execute('INSERT INTO parts VALUES (?,?,?,?)', (p['path'], p['sha256'], p['size'], json.dumps(detail)))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        progress({'part': p['path'], 'status': 'INDEXED_COMMITTED', 'stats': detail})
        path.unlink()
        print(json.dumps({'part': p['path'], 'status': 'INDEXED_COMMITTED', 'retained_records': stats['selected_language_records']}), flush=True)
    iterator = iter(todo)
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(fetch, p): p for p in [next(iterator, None) for _ in range(min(8, len(todo)))] if p}
        while pending:
            complete, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for future in complete:
                pending.pop(future)
                p, path = future.result()
                ingest(p, path)
                next_part = next(iterator, None)
                if next_part:
                    pending[pool.submit(fetch, next_part)] = next_part
    summaries = [{'part': r[0], 'sha256': r[1], 'size': r[2], 'stats': json.loads(r[3])} for r in conn.execute('SELECT name,sha256,size,stats FROM parts ORDER BY name')]
    assert len(summaries) == 20
    counts = {r[0]: r[1] for r in conn.execute('SELECT lang,COUNT(*) FROM raw GROUP BY lang')}
    coverage = {lang: conn.execute("SELECT COUNT(*) FROM raw t JOIN raw e ON e.lang='en' AND e.record_id=t.record_id WHERE t.lang=?", (lang,)).fetchone()[0] for lang in ['sw', 'bn', 'ja']}
    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    conn.close()
    dump(out / 'RAW_INDEX_DONE.json', {'status': 'COMPLETE', 'dataset_revision': REV, 'parts': summaries, 'retained_language_counts': counts, 'counter_matched_english_records': coverage, 'counter_is_not_original_source_group': True, 'raw_shard_bytes': sum((x['size'] for x in summaries)), 'temporary_shards_removed_only_after_committed_verified_index': True, 'database_file': 'raw_selected_languages.sqlite3', 'sampling_training_and_semantic_independence': 'NOT_RUN; original source grouping and data/token checks still required before any training.'})
    print(json.dumps({'status': 'COMPLETE_RAW_INDEX_ONLY', 'retained_counts': counts, 'english_counter_coverage': coverage}), flush=True)
if __name__ == '__main__':
    main()
