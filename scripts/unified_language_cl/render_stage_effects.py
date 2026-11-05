"""Render all seven completed task matrices, keeping task units separate."""
from runtime import runtime_path
import argparse
import csv
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, sha, dump
TRACKS = ['massive', 'mathoctopus', 'xnli', 'mtop', 'xlsum', 'wikilingua', 'sib200']
LABELS = {'massive': 'MASSIVE', 'mathoctopus': 'MathOctopus', 'xnli': 'XNLI', 'mtop': 'MTOP', 'xlsum': 'XL-Sum', 'wikilingua': 'WikiLingua', 'sib200': 'SIB-200'}
MODELS = ['base', 'single_es', 'single_th', 'seq_es_th']
RETENTION = [('reading:acc:none', 'EN Belebele ACC'), ('gsm8k:exact_match:flexible-extract', 'GSM8K flexible ACC'), ('ifeval:prompt_level_strict_acc:none', 'IFEval prompt strict'), ('ifeval:inst_level_strict_acc:none', 'IFEval instruction strict')]

def summary_provenance(track, effect):
    root = Path(runtime_path('results', '')) / '' / track
    aliases = json.loads((root / 'run_aliases.json').read_text()) if (root / 'run_aliases.json').exists() else {}
    for model in MODELS:
        folder = root / aliases.get(model + '-test-v1', model + '-test-v1')
        run = json.loads((folder / 'run.json').read_text())
        assert run['primary_scorer_version'] == 'v2_explicit_multilingual_sentence_tokenizer'
        assert run['score_correction_protocol_sha256'] == sha(Path(runtime_path('configs', '')) / 'SUMMARY_SCORING_CORRECTION_v2.json')
        assert effect['sources'][model]['run_sha256'] == sha(folder / 'run.json')

def load_all():
    native, retention, sources = ({}, {}, {})
    for track in TRACKS:
        base = Path(runtime_path('results', '')) / '' / track
        for scope in ['comparison-test-v1', 'retention-comparison-v1']:
            path = base / scope / 'effects.json'
            assert path.exists(), 'INCOMPLETE: ' + track + '/' + scope
            value = json.loads(path.read_text())
            if scope == 'comparison-test-v1':
                assert value['status'] == 'COMPLETE_FIXED_FINAL_MATRIX'
                assert value['same_prompts_references_ids_verified']
                if track in ['xlsum', 'wikilingua']:
                    summary_provenance(track, value)
                native[track] = value
            else:
                assert value['same_docs_prompts_targets_verified']
                retention[track] = value
            sources[track + '/' + scope] = sha(path)
    return (native, retention, sources)

def interval(axis, effect, y=0, color='#245b89'):
    lo, hi, point = [effect[key] for key in ['ci95_low', 'ci95_high', 'delta']]
    axis.hlines(y, lo, hi, color=color, linewidth=2)
    axis.vlines([lo, hi], y - 0.09, y + 0.09, color=color, linewidth=1)
    axis.scatter([point], [y], color=color, s=22, zorder=3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    native, retention, sources = load_all()
    assert not args.output.exists(), 'Choose a new report namespace'
    args.output.mkdir(parents=True)
    records, effects, panels = ([], [], [])
    for track in TRACKS:
        groups = native[track]['groups']
        benchmarks = sorted({key.split(':', 1)[1] for key in groups})
        for benchmark in benchmarks:
            for language in ['es', 'th', 'fr', 'en']:
                value = groups[language + ':' + benchmark]
                metric = 'ROUGE-Lsum F1 points' if track in ['xlsum', 'wikilingua'] else 'Tree EM percentage points' if track == 'mtop' else 'ACC percentage points'
                records.append({'track': track, 'benchmark': benchmark, 'language': language, 'n': value['samples'], 'metric': metric, **value['scores']})
                for name, e in value['effects'].items():
                    effects.append({'track': track, 'benchmark': benchmark, 'language': language, 'effect': name, 'metric': metric, **e})
            panels.append((track, benchmark, groups['es:' + benchmark], groups['th:' + benchmark]))
    columns = [('es', 'single_es_minus_base', 'ES learning\nW1 - Base'), ('th', 'single_es_minus_base', 'Transfer to unseen TH\nW1 - Base'), ('th', 'seq_th_minus_es', 'TH stage gain\nW2 - W1'), ('es', 'es_forgetting', 'ES forgetting\nW1 - W2')]
    figure, axes = plt.subplots(len(panels), len(columns), figsize=(15, 2 * len(panels)), squeeze=False)
    for row, (track, benchmark, es, th) in enumerate(panels):
        unit = 'ROUGE-Lsum points' if track in ['xlsum', 'wikilingua'] else 'Tree EM pp' if track == 'mtop' else 'ACC pp'
        label = LABELS[track] + ('\n' + benchmark if track == 'mathoctopus' else '') + '\n' + unit
        for col, (language, effect, title) in enumerate(columns):
            axis = axes[row, col]
            value = (es if language == 'es' else th)['effects'][effect]
            interval(axis, value)
            axis.axvline(0, color='#999999', linewidth=0.7, linestyle='--')
            low, high = (min(0, value['ci95_low']), max(0, value['ci95_high']))
            span = max(high - low, 1)
            axis.set_xlim(low - 0.15 * span, high + 0.15 * span)
            axis.set_ylim(-0.45, 0.7)
            axis.set_yticks([])
            axis.text(0.5, 0.91, f"{value['delta']:+.2f} [{value['ci95_low']:+.2f}, {value['ci95_high']:+.2f}]", transform=axis.transAxes, ha='center', va='top', fontsize=8)
            axis.spines[['left', 'top', 'right']].set_visible(False)
            axis.tick_params(axis='x', labelsize=8)
            if col == 0:
                axis.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=12, fontsize=9)
            if row == 0:
                axis.set_title(title, fontsize=10, pad=12)
    figure.suptitle('Fixed ES -> TH Finals: learning, transfer and forgetting', fontsize=13)
    figure.text(0.5, 0.01, 'Paired 95% source-unit bootstrap, conditional on seed42 models. No training-seed uncertainty or multiplicity correction.\nEach panel retains its task unit and own axis scale; no cross-task average. Positive ES forgetting means decline.', ha='center', fontsize=8)
    figure.tight_layout(rect=(0.02, 0.04, 1, 0.97))
    for suffix in ['png', 'pdf']:
        figure.savefig(args.output / ('native_stage_effects.' + suffix), dpi=200, bbox_inches='tight')
    plt.close(figure)
    figure, axes = plt.subplots(1, len(RETENTION), figsize=(17, 5), sharey=True)
    for col, (key, title) in enumerate(RETENTION):
        axis = axes[col]
        values = [retention[track]['groups'][key]['effects']['seq_final_minus_base'] for track in TRACKS]
        for index, value in enumerate(values):
            interval(axis, value, y=len(TRACKS) - index - 1)
        axis.axvline(0, color='#999999', linewidth=0.8, linestyle='--')
        axis.set_yticks(range(len(TRACKS)), [LABELS[t] for t in reversed(TRACKS)])
        axis.set_title(title, fontsize=11)
        axis.set_xlabel('SEQ Final - Base (percentage points)')
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('General-ability changes under the same frozen evaluation protocol', fontsize=12)
    figure.text(0.5, 0.01, 'Fixed Finals; no EN early stopping. Conditional paired 95% intervals; IFEval instructions remain clustered by prompt in source tables.', ha='center', fontsize=8)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    for suffix in ['png', 'pdf']:
        figure.savefig(args.output / ('general_ability_changes.' + suffix), dpi=200, bbox_inches='tight')
    plt.close(figure)
    for name, values in [('native_scores.csv', records), ('native_effects.csv', effects)]:
        fields = sorted({key for row in values for key in row})
        with (args.output / name).open('x') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    dump(args.output / 'manifest.json', {'status': 'ALL_SEVEN_MAIN_MATRICES_RENDERED', 'source_effects_sha256': sources, 'models': MODELS, 'tracks': TRACKS, 'native_groups': len(records), 'summary_correction_policy_sha256': sha(Path(runtime_path('configs', '')) / 'SUMMARY_SCORING_CORRECTION_v2.json'), 'limits': ['Fixed model conditional intervals, not repeated-seed variance.', 'Native task units are kept separate; do not average ACC and ROUGE.', 'Summary primary scorer v2 provenance verified.', 'These are main matrices; P3 cohort controls are reported separately.']})
    print(json.dumps({'status': 'ALL_SEVEN_MAIN_MATRICES_RENDERED', 'native_groups': len(records), 'output': str(args.output)}))
if __name__ == '__main__':
    main()
