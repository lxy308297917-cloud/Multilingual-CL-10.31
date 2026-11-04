"""Paired task-unit effects from fixed Finals; no score-based model selection."""
from runtime import runtime_path
import argparse, collections, json, os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from common import ROOT, DATA, sha, oh, dump, rows
from evaluate import metrics
MODELS = ['base', 'single_es', 'single_th', 'seq_es_th']

def bootstrap(matrix, clusters, contrast, repeats=2000):
    diff = np.asarray(matrix, dtype=float) @ np.asarray(contrast, dtype=float)
    groups = collections.defaultdict(list)
    for i, k in enumerate(clusters):
        groups[k].append(i)
    keys = sorted(groups)
    sums = np.array([sum(diff[groups[k]]) for k in keys])
    ns = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(42)
    samples = []
    for off in range(0, repeats, 128):
        ix = rng.integers(0, len(keys), size=(min(128, repeats - off), len(keys)))
        samples.extend((sums[ix].sum(1) / ns[ix].sum(1)).tolist())
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {'delta': float(diff.mean()), 'ci95_low': float(lo), 'ci95_high': float(hi), 'repeats': repeats, 'n': len(diff), 'clusters': len(keys), 'unit': 'conditional paired source-ID bootstrap; single training seed, no multiplicity correction'}

def load(path, summary=False):
    rs = rows(path / 'predictions.jsonl')
    if summary:
        vals = {r['id']: r for r in rows(path / 'per_sample_scores.jsonl')}
        original = (path / 'predictions.jsonl').resolve().parent / 'per_sample_scores.jsonl'
        original_vals = {r['id']: r for r in rows(original)}
        assert set(vals) == set(original_vals)
        for id, v in vals.items():
            assert all((abs(v[m] - original_vals[id][m]) < 1e-08 for m in ['rouge1', 'rouge2'])), ('ROUGE1/2 per-row regression', id)
        for r in rs:
            r['outcome'] = vals[r['id']]['rougeLsum']
    else:
        metrics(rs)
        for r in rs:
            r['outcome'] = float(r['correct']) * 100
    return rs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', required=True)
    ap.add_argument('--split', choices=['dev', 'test'], default='test')
    a = ap.parse_args()
    root = Path(runtime_path('results', '')) / '' / a.track
    models = {}
    source = {}
    for name in MODELS:
        suffix = name + '-' + a.split + '-v1'
        if name == 'base' and a.track == 'mtop' and (a.split == 'dev'):
            suffix = 'base-dev-v2-syntax'
        aliases = json.loads((root / 'run_aliases.json').read_text()) if (root / 'run_aliases.json').exists() else {}
        suffix = aliases.get(suffix, suffix)
        assert Path(suffix).name == suffix, 'Aliases must stay inside track results'
        p = root / suffix
        if not (p / 'DONE.json').exists():
            print(json.dumps({'status': 'INCOMPLETE', 'missing': str(p)}))
            return
        if a.track in ['xlsum', 'wikilingua']:
            run = json.loads((p / 'run.json').read_text())
            assert run['primary_scorer_version'] == 'v2_explicit_multilingual_sentence_tokenizer'
            assert run['score_correction_protocol_sha256'] == sha(Path(runtime_path('configs', '')) / 'SUMMARY_SCORING_CORRECTION_v2.json')
        models[name] = load(p, a.track in ['xlsum', 'wikilingua'])
        source[name] = {'predictions_sha256': sha(p / 'predictions.jsonl'), 'run_sha256': sha(p / 'run.json'), 'scores_sha256': sha(p / 'scores.json')}
    wiki_clusters = json.loads((DATA / 'prepared-v1/wikilingua/source_clusters.json').read_text()) if a.track == 'wikilingua' else {}
    if a.track == 'wikilingua':
        policy = json.loads((Path(runtime_path('configs', '')) / 'WIKILINGUA_CLUSTER_POLICY_v1.json').read_text())
        assert sha(DATA / 'prepared-v1/wikilingua/source_clusters.json') == policy['mapping_sha256']
    groups = collections.defaultdict(list)
    for r in models['base']:
        groups[r['lang'], r.get('benchmark', a.track)].append(r['id'])
    maps = {name: {(r['lang'], r.get('benchmark', a.track), r['id']): r for r in rs} for name, rs in models.items()}
    result = {}
    matrix_rows = []
    for (lang, benchmark), ids in sorted(groups.items()):
        matrix = []
        clusters = []
        for id in ids:
            cells = [maps[name][lang, benchmark, id] for name in MODELS]
            assert len({oh([r['prompt'], r['answer'], r.get('gold_numeric')]) for r in cells}) == 1
            assert len({r['prompt_token_sha256'] for r in cells}) == 1
            matrix.append([r['outcome'] for r in cells])
            clusters.append(wiki_clusters[id] if a.track == 'wikilingua' else cells[0].get('semantic_group', id))
        arr = np.array(matrix)
        scores = dict(zip(MODELS, map(float, arr.mean(0))))
        effects = {}
        for label, c in [('single_es_minus_base', [-1, 1, 0, 0]), ('single_th_minus_base', [-1, 0, 1, 0]), ('seq_th_minus_es', [0, -1, 0, 1]), ('es_forgetting', [0, 1, 0, -1]), ('seq_final_minus_single_th', [0, 0, -1, 1]), ('difference_of_th_gains', [1, -1, -1, 1])]:
            effects[label] = bootstrap(arr, clusters, c)
        if a.track == 'wikilingua':
            for effect in effects.values():
                effect['unit'] = 'conditional paired English-article cluster bootstrap; sections within article grouped; seed42 only'
        key = lang + ':' + benchmark
        result[key] = {'samples': len(ids), 'metric': 'ROUGE-Lsum F1 points' if a.track in ['xlsum', 'wikilingua'] else 'ACC percentage points', 'scores': scores, 'effects': effects}
        matrix_rows.append('| ' + benchmark + ' | ' + lang + ' | ' + str(len(ids)) + ' | ' + ' | '.join((f'{scores[n]:.2f}' for n in MODELS)) + ' |')
    report = {'track': a.track, 'split': a.split, 'status': 'COMPLETE_FIXED_FINAL_MATRIX', 'model_order': MODELS, 'groups': result, 'sources': source, 'same_prompts_references_ids_verified': True, 'limits': ['Only seed42; bootstrap conditions on these trained models.', 'No multiplicity adjustment; interval crossing zero is not equivalence.', 'No pooling unlike task metrics or selecting checkpoints from tests.', 'Native classification generation can improve via label/format acquisition; inspect matched dev64 candidate diagnosis before claiming new language semantics.']}
    out = root / ('comparison-' + a.split + '-v1')
    assert not out.exists()
    out.mkdir()
    dump(out / 'effects.json', report)
    text = ['# ' + a.track + '：固定Final阶段矩阵', '', '| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |', '|---|---|---:|---:|---:|---:|---:|'] + matrix_rows + ['', '单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。', '', '| Benchmark | 指标 | 变化及95%区间 |', '|---|---|---:|']
    for key, g in result.items():
        lang, benchmark = key.split(':', 1)
        wanted = [('G_ES', 'single_es_minus_base'), ('F_ES', 'es_forgetting')] if lang == 'es' else [('Single TH收益', 'single_th_minus_base'), ('G_TH', 'seq_th_minus_es'), ('T_TH', 'single_es_minus_base'), ('SEQ Final−Single TH Final', 'seq_final_minus_single_th'), ('阶段收益差（非因果）', 'difference_of_th_gains')] if lang == 'th' else [('ES阶段迁移', 'single_es_minus_base'), ('TH阶段变化', 'seq_th_minus_es')]
        for label, e in wanted:
            v = g['effects'][e]
            text.append(f"| {key} | {label} | {v['delta']:+.2f} [{v['ci95_low']:+.2f}, {v['ci95_high']:+.2f}] |")
    (out / 'table.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({'status': report['status'], 'track': a.track, 'split': a.split, 'groups': len(result)}))
if __name__ == '__main__':
    main()
