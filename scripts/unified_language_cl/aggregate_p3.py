"""Fixed six-model Same-ID/Disjoint-ID contrasts, without test-based selection."""
from runtime import runtime_path
import argparse
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from common import ROOT, DATA, MODEL, sha, oh, dump, rows
from make_configs import CKPT
from aggregate import bootstrap
from evaluate import metrics
from aggregate_retention import read as read_retention, directory as retention_directory
MODELS = ['base', 'single_es', 'single_th', 'seq_es_th', 'disjoint_single_th', 'disjoint_seq_es_th']
CONTRASTS = {'same_id_seq_minus_direct_th': [0, 0, -1, 1, 0, 0], 'disjoint_id_seq_minus_direct_th': [0, 0, 0, 0, -1, 1], 'same_minus_disjoint_seq_advantage': [0, 0, -1, 1, 1, -1], 'same_id_es_forgetting': [0, 1, 0, -1, 0, 0], 'disjoint_id_es_forgetting': [0, 1, 0, 0, 0, -1], 'same_id_th_stage_change': [0, -1, 0, 1, 0, 0], 'disjoint_id_th_stage_change': [0, -1, 0, 0, 0, 1], 'direct_th_A_minus_B': [0, 0, 1, 0, -1, 0]}

def summarize(matrix, clusters):
    scores = dict(zip(MODELS, map(float, np.asarray(matrix).mean(0))))
    return {'scores': scores, 'n': len(matrix), 'clusters': len(set(clusters)), 'effects': {name: bootstrap(matrix, clusters, weights) for name, weights in CONTRASTS.items()}}

def native(track, split):
    root = Path(runtime_path('results', '')) / '' / track
    aliases = json.loads((root / 'run_aliases.json').read_text()) if (root / 'run_aliases.json').exists() else {}
    datasets, sources = ({}, {})
    parent_sha = sha(DATA / 'prepared-v1' / track / 'protocol.json')
    protocol = json.loads((DATA / 'prepared-v1' / track / 'protocol.json').read_text())
    for model in MODELS:
        suffix = model + '-' + split + '-v1'
        suffix = aliases.get(suffix, suffix)
        assert Path(suffix).name == suffix
        folder = root / suffix
        assert (folder / 'DONE.json').exists(), folder
        run = json.loads((folder / 'run.json').read_text())
        assert run['protocol_sha256'] == parent_sha, 'Evaluate original unchanged native dev/test, not B training data'
        expected_model = MODEL if model == 'base' else CKPT / (track + '_' + model + '_seed42_v1')
        assert Path(run['model']).resolve() == expected_model.resolve(), 'Wrong model role'
        if model != 'base':
            validation = json.loads((expected_model / 'VALIDATED.json').read_text())
            assert validation['full_epoch_exact_multiset']
            assert run['weights_sha256'] == validation['weights']
        rs = rows(folder / 'predictions.jsonl')
        completion = json.loads((folder / 'DONE.json').read_text())
        assert completion['predictions_sha256'] == sha(folder / 'predictions.jsonl')
        for lang in ['es', 'th', 'fr', 'en']:
            assert sum((r['lang'] == lang for r in rs)) == protocol['token_stats'][lang + ':' + split]['n'], 'Incomplete native evaluation'
        metrics(rs)
        datasets[model] = {(r['lang'], r.get('benchmark', track), r['id']): r for r in rs}
        assert len(datasets[model]) == len(rs)
        sources[model] = {name: sha(folder / name) for name in ['run.json', 'predictions.jsonl', 'scores.json']}
    keys = set(datasets['base'])
    assert all((set(values) == keys for values in datasets.values()))
    grouped = {}
    for key in sorted(keys):
        cells = [datasets[model][key] for model in MODELS]
        assert len({oh([r['prompt'], r['answer'], r.get('gold_numeric')]) for r in cells}) == 1
        assert len({r['prompt_token_sha256'] for r in cells}) == 1
        group = grouped.setdefault(':'.join(key[:2]), {'matrix': [], 'clusters': []})
        group['matrix'].append([100 * float(r['correct']) for r in cells])
        group['clusters'].append(cells[0].get('semantic_group', key[2]))
    return ({key: summarize(**value) for key, value in grouped.items()}, sources)

def retention(track):
    datasets, sources = ({}, {})
    for model in MODELS:
        if model.startswith('disjoint_'):
            folder = Path(runtime_path('results', '')) / 'retention' / (track + '_' + model) / 'formal-v1'
        else:
            folder = retention_directory(track, model)
        if model != 'base':
            validation = json.loads((CKPT / (track + '_' + model + '_seed42_v1') / 'VALIDATED.json').read_text())
            assert validation['full_epoch_exact_multiset']
            assert json.loads((folder / 'DONE.json').read_text())['weights_sha256'] == validation['weights']
        datasets[model], sources[model] = read_retention(folder)
    keys = set(datasets['base'])
    assert all((set(value) == keys for value in datasets.values()))
    groups = {}
    for key in sorted(keys):
        ids = sorted(datasets['base'][key])
        expected = 900 if key.startswith('reading:') else 1319 if key.startswith('gsm8k:') else 541
        assert len(ids) == expected
        assert all((set(value[key]) == set(ids) for value in datasets.values()))
        matrix, clusters = ([], [])
        for identifier in ids:
            cells = [datasets[model][key][identifier] for model in MODELS]
            assert len({(r['doc_hash'], r['prompt_hash'], r['target_hash']) for r in cells}) == 1
            assert len({len(r['outcomes']) for r in cells}) == 1
            for index in range(len(cells[0]['outcomes'])):
                matrix.append([r['outcomes'][index] for r in cells])
                clusters.append(str(identifier))
        groups[key] = summarize(matrix, clusters)
        if 'inst_level_' in key:
            assert len(matrix) == 834
    return (groups, sources)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', choices=['massive', 'xnli'], required=True)
    parser.add_argument('--scope', choices=['dev', 'test', 'retention'], required=True)
    args = parser.parse_args()
    parent = json.loads((DATA / 'prepared-v1' / args.track / 'protocol.json').read_text())
    derived = json.loads((DATA / 'prepared-disjoint-v1' / args.track / 'protocol.json').read_text())
    assert derived['training_ids'] == parent['disjoint_reserve_ids']
    assert len(derived['training_ids']) == 3200 and (not set(derived['training_ids']) & set(parent['training_ids']))
    plan = Path(runtime_path('configs', '')) / 'P3_ANALYSIS_PLAN_v1.json'
    assert json.loads(plan.read_text())['contrasts'] == CONTRASTS
    groups, sources = retention(args.track) if args.scope == 'retention' else native(args.track, args.scope)
    out = Path(runtime_path('results', '')) / '' / args.track / ('p3-' + args.scope + '-comparison-v1')
    assert not out.exists()
    out.mkdir()
    dump(out / 'effects.json', {'track': args.track, 'scope': args.scope, 'model_order': MODELS, 'analysis_plan_sha256': sha(plan), 'groups': groups, 'sources': sources, 'status': 'COMPLETE_FIXED_P3_COMPARISON', 'limits': ['Conditional paired 2000-repeat source-ID/prompt bootstrap, seed42 only; no training-seed uncertainty or multiplicity correction.', 'A and B are different predeclared training cohorts, with potentially different difficulty and label composition.', 'The advantage interaction is not a pure language-switch causal effect or independent replication.', 'MASSIVE contains exact normalized utterance overlap despite disjoint IDs; report its separate audit.', 'Do not pool unlike metrics or select checkpoints/configurations from these tests.']})
    table = [f'# {args.track}：Same-ID/Disjoint-ID {args.scope}', '', '单位：ACC/通过率百分点；固定Final。', '', '| 指标 | n | Base | ES | TH_A | ES→TH_A | TH_B | ES→TH_B |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, value in groups.items():
        if 'loose' not in key:
            table.append('| ' + key + ' | ' + str(value['n']) + ' | ' + ' | '.join((f"{value['scores'][model]:.2f}" for model in MODELS)) + ' |')
    table += ['', '| 指标 | (SEQ_A−TH_A)−(SEQ_B−TH_B)，95%区间 |', '|---|---:|']
    for key, value in groups.items():
        if 'loose' not in key:
            e = value['effects']['same_minus_disjoint_seq_advantage']
            table.append(f"| {key} | {e['delta']:+.2f} [{e['ci95_low']:+.2f}, {e['ci95_high']:+.2f}] |")
    (out / 'table.md').write_text('\n'.join(table) + '\n')
    print(json.dumps({'status': 'COMPLETE_FIXED_P3_COMPARISON', 'track': args.track, 'scope': args.scope, 'metrics': len(groups)}))
if __name__ == '__main__':
    main()
