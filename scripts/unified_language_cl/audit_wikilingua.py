"""Inspect verified author-supported GEM splits without executing pickle globals."""
from runtime import runtime_path
import os, sys, json, pickle, tarfile, collections, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, runtime_path('project', 'scripts/unified_language_cl'))
from common import DATA, ROOT, sha, oh, norm, dump, write_rows
import pyarrow.parquet as pq

class PrimitiveUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        raise pickle.UnpicklingError('Forbidden global ' + module + '.' + name)

def main():
    out = DATA / 'gem-wikilingua-audit-v2'
    assert not out.exists()
    out.mkdir()
    index = {}
    raw_counts = {}
    files = {}
    for lang, name in [('es', 'spanish'), ('th', 'thai'), ('fr', 'french'), ('en', 'english')]:
        f = DATA / f'sources/esdurmus--wiki_lingua/{name}/train-00000-of-00001.parquet'
        lookup = collections.defaultdict(set)
        total = 0
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
            for r in batch.to_pylist():
                a = r['article']
                for i, (x, y) in enumerate(zip(a['document'], a['summary'])):
                    url = r['url'] if lang == 'en' else a['english_url'][i]
                    section = a['section_name'][i] if lang == 'en' else a['english_section_name'][i]
                    lookup[oh([norm(x), norm(y)])].add((oh([url, section]), url))
                    total += 1
        index[lang] = lookup
        raw_counts[lang] = total
        files[str(f)] = sha(f)
        print(json.dumps({'raw_index': lang, 'sections': total}), flush=True)
    archive = DATA / 'sources/GEM--wiki_lingua/wikilingua_cleaned.tar.gz'
    verification = json.loads(archive.with_name('download_verified.json').read_text())
    all_rows = {}
    counts = {}
    unmatched = {}
    ambiguous = {}
    seen_members = []
    wanted = {f'cleaned/wikilingua_multilingual.{s}.pk': 'dev' if s == 'val' else s for s in ['train', 'val', 'test']}
    with tarfile.open(archive, 'r|gz') as tf:
        for m in tf:
            if m.name not in wanted:
                continue
            assert m.isfile()
            split = wanted[m.name]
            seen_members.append(m.name)
            data = PrimitiveUnpickler(tf.extractfile(m)).load()
            assert isinstance(data, list)
            rs = []
            ct = collections.Counter()
            bad = collections.Counter()
            amb = collections.Counter()
            for d in data:
                assert isinstance(d, dict)
                l = d['source']
                if l != d['target'] or l not in index:
                    continue
                x = d['document'].strip()
                y = d['summary'].strip()
                matches = index[l].get(oh([norm(x), norm(y)]), set())
                ct[l] += 1
                if not x or not y:
                    continue
                if len(matches) != 1:
                    (bad if not matches else amb)[l] += 1
                    continue
                sid, url = next(iter(matches))
                rs.append({'id': str(d['id']), 'lang': l, 'split': split, 'semantic_id': sid, 'english_article_url': url, 'text': x, 'answer': y})
            all_rows[split] = rs
            counts[split] = dict(ct)
            unmatched[split] = dict(bad)
            ambiguous[split] = dict(amb)
            write_rows(out / (split + '.jsonl'), rs)
            print(json.dumps({'split': split, 'native_counts': dict(ct), 'uniquely_mapped': len(rs), 'unmatched': dict(bad), 'ambiguous': dict(amb)}), flush=True)
            del data
    assert set(seen_members) == set(wanted)
    articles = {s: {r['english_article_url'] for r in rs} for s, rs in all_rows.items()}
    ids = {s: {r['semantic_id'] for r in rs} for s, rs in all_rows.items()}
    overlap = {a + '_' + b: {'article_urls': len(articles[a] & articles[b]), 'section_ids': len(ids[a] & ids[b])} for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]}
    hold = articles['dev'] | articles['test']
    eligible = [r for r in all_rows['train'] if r['english_article_url'] not in hold]
    bylang = {l: {r['semantic_id'] for r in eligible if r['lang'] == l} for l in index}
    common = set.intersection(*bylang.values())
    common3 = set.intersection(*(bylang[l] for l in ['es', 'th', 'fr']))
    report = {'status': 'GEM_SPLITS_AUDITED_PENDING_PREPARATION', 'revision': 'af5d0f00b59a6933165c97b384f50d8b563c314d', 'archive_verification': verification, 'counts': counts, 'raw_section_counts': raw_counts, 'raw_hashes': files, 'unmatched': unmatched, 'ambiguous': ambiguous, 'split_overlap': overlap, 'eligible_train_counts': {l: sum((r['lang'] == l for r in eligible)) for l in index}, 'common4_eligible_train_ids': len(common), 'common3_eligible_train_ids': len(common3), 'mapping': 'Exact normalized document+summary to publisher English URL+section fields; ambiguous or unmatched records excluded from strict mapping. Author split rows are never aligned by generic ID or position.', 'license': 'CC-BY-NC-SA-3.0', 'article_split_policy': 'Exclude training English article URLs occurring in dev/test in ANY mapped language. Do not modify evaluation splits.'}
    dump(out / 'audit.json', report)
    dump(Path(runtime_path('results', '')) / 'wikilingua/gem_raw_audit_v1.json', report)
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
