"""Paired fixed-protocol old-ability changes, instruction outcomes clustered by prompt."""
from runtime import runtime_path
import argparse, json, collections, sys, os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from common import ROOT, sha, dump, rows
from aggregate import bootstrap
MODELS = ['base', 'single_es', 'single_th', 'seq_es_th']

def directory(track, name):
    r = Path(runtime_path('results', '')) / 'retention'
    key = 'base' if name == 'base' else track + '_' + name
    return r / key / ('formal-v2-auditfix' if name == 'base' or track in ['massive', 'mathoctopus'] else 'formal-v1')

def read(folder):
    assert (folder / 'DONE.json').exists(), folder
    records = {}
    sources = {}
    for scope in ['en_reading', 'en_core']:
        base = folder / scope / 'results'
        fns = list(base.rglob('samples_*.jsonl'))
        assert fns
        for f in fns:
            source_rows = rows(f)
            assert source_rows
            sources[str(f.relative_to(folder))] = sha(f)
            for r in source_rows:
                task = 'reading' if scope == 'en_reading' else 'gsm8k' if 'gsm8k' in f.name else 'ifeval'
                filter = r['filter']
                metrics = ['acc'] if task == 'reading' else list(r['metrics'])
                for metric in metrics:
                    if task == 'ifeval' and metric not in ['prompt_level_strict_acc', 'inst_level_strict_acc', 'prompt_level_loose_acc', 'inst_level_loose_acc']:
                        continue
                    key = task + ':' + metric + ':' + filter
                    v = r[metric]
                    vals = v if isinstance(v, list) else [v]
                    assert all((x in [0, 1] for x in vals))
                    records.setdefault(key, {})[r['doc_id']] = {'outcomes': [100 * float(x) for x in vals], 'doc_hash': r['doc_hash'], 'prompt_hash': r['prompt_hash'], 'target_hash': r['target_hash']}
    return (records, sources)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', required=True)
    a = ap.parse_args()
    datasets = {}
    sources = {}
    for name in MODELS:
        datasets[name], sources[name] = read(directory(a.track, name))
    assert all((set(datasets[n]) == set(datasets['base']) for n in MODELS))
    groups = {}
    table = ['# ' + a.track + '：统一旧能力监测', '', '单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。', '', '| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |', '|---|---:|---:|---:|---:|---:|']
    effects_table = ['', '| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |', '|---|---:|---:|']
    for key in sorted(datasets['base']):
        ids = sorted(datasets['base'][key])
        assert all((set(datasets[n][key]) == set(ids) for n in MODELS))
        matrix = []
        clusters = []
        for id in ids:
            cells = [datasets[n][key][id] for n in MODELS]
            assert len({(r['doc_hash'], r['prompt_hash'], r['target_hash']) for r in cells}) == 1, (key, id, 'input mismatch')
            assert len({len(r['outcomes']) for r in cells}) == 1
            for j in range(len(cells[0]['outcomes'])):
                matrix.append([c['outcomes'][j] for c in cells])
                clusters.append(str(id))
        n = len(matrix)
        scores = {name: sum((row[i] for row in matrix)) / n for i, name in enumerate(MODELS)}
        effects = {}
        for label, c in [('single_es_minus_base', [-1, 1, 0, 0]), ('single_th_minus_base', [-1, 0, 1, 0]), ('seq_th_minus_es', [0, -1, 0, 1]), ('seq_final_minus_base', [-1, 0, 0, 1])]:
            effects[label] = bootstrap(matrix, clusters, c)
        groups[key] = {'samples_or_instructions': n, 'prompts': len(ids), 'scores': scores, 'effects': effects, 'unit': 'ACC / pass-rate percentage points; instruction outcomes clustered by prompt'}
        if 'loose' not in key:
            table.append('| ' + key + ' | ' + str(n) + ' | ' + ' | '.join((f'{scores[n]:.2f}' for n in MODELS)) + ' |')

            def fmt(e):
                return f"{e['delta']:+.2f} [{e['ci95_low']:+.2f}, {e['ci95_high']:+.2f}]"
            effects_table.append('| ' + key + ' | ' + fmt(effects['seq_th_minus_es']) + ' | ' + fmt(effects['seq_final_minus_base']) + ' |')
    out = Path(runtime_path('results', '')) / f'{a.track}/retention-comparison-v1'
    assert not out.exists()
    out.mkdir()
    dump(out / 'effects.json', {'track': a.track, 'model_order': MODELS, 'groups': groups, 'sources': sources, 'same_docs_prompts_targets_verified': True, 'limits': ['Single training seed42; CIs condition on trained models, no multiplicity correction.', 'Do not pool unrelated metrics into an average forgetting score.', 'GSM8K strict decline can include format loss; report flexible-extract alongside it.', 'IFEval instruction intervals resample prompts to retain within-prompt dependence.']})
    (out / 'table.md').write_text('\n'.join(table + effects_table) + '\n')
    print(json.dumps({'status': 'DONE', 'track': a.track, 'metrics': len(groups)}))
if __name__ == '__main__':
    main()
