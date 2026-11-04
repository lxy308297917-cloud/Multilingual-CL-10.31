"""Freeze article-cluster metadata before WikiLingua inference; no prompt/label changes."""
from runtime import runtime_path
import sys, json
from pathlib import Path
sys.path.insert(0, runtime_path('project', 'scripts/unified_language_cl'))
from common import DATA, ROOT, rows, oh, sha, dump
raw = DATA / 'gem-wikilingua-audit-v2'
prepared = DATA / 'prepared-v1/wikilingua'
mapping = {}
for split in ['train', 'dev', 'test']:
    source = {(r['lang'], r['semantic_id']): oh(r['english_article_url']) for r in rows(raw / (split + '.jsonl'))}
    for l in ['es', 'th', 'fr', 'en']:
        for r in rows(prepared / (l + '_' + split + '.jsonl')):
            sid = r['id'].split(':', 1)[1]
            value = source[l, sid]
            assert r['id'] not in mapping or mapping[r['id']] == value
            mapping[r['id']] = value
f = prepared / 'source_clusters.json'
assert not f.exists()
dump(f, mapping)
dump(Path(runtime_path('configs', '')) / 'WIKILINGUA_CLUSTER_POLICY_v1.json', {'frozen_before_wiki_inference': True, 'unit': 'publisher English article URL digest, paired across models; language-specific estimates', 'mapping_sha256': sha(f), 'rows': len(mapping), 'article_clusters': len(set(mapping.values())), 'prompt_or_label_changes': False, 'rationale': 'Multiple sections share articles. Official section train/dev/test may share an article; training excludes mapped evaluation articles, and inferential intervals cluster sections by original article.'})
print('Article cluster metadata frozen', len(mapping))
