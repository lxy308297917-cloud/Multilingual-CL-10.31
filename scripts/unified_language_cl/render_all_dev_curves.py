"""Seven-track fixed dev curves; no new generation or checkpoint selection."""
from runtime import runtime_path
import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, runtime_path('project', 'scripts/unified_language_cl'))
from common import ROOT, DATA, MODEL, sha, oh, rows, dump
from make_configs import CKPT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
TRACKS = ['massive', 'mathoctopus', 'mtop', 'xnli', 'xlsum', 'wikilingua', 'sib200']
LANGS = ['es', 'th', 'fr', 'en']

def collect():
    data, provenance = ([], {})
    for track in TRACKS:
        root = Path(runtime_path('results', '')) / '' / track
        parent = DATA / ('prepared-v2' if track == 'mtop' else 'prepared-v1') / track
        protocol = json.loads((parent / 'protocol.json').read_text())
        final, mid = (protocol['steps'], protocol['steps'] // 2)
        aliases = json.loads((root / 'run_aliases.json').read_text()) if (root / 'run_aliases.json').exists() else {}
        roles = [('base', 0, 'base-dev-v2-syntax' if track == 'mtop' else 'base-dev-v1')]
        roles += [(arm, step, f'{arm}-dev{mid}-v1' if step == mid else f'{arm}-dev-v1') for arm in ['single_es', 'single_th', 'seq_es_th'] for step in [mid, final]]
        for _, _, role in roles:
            assert (root / aliases.get(role, role) / 'DONE.json').exists(), ('INCOMPLETE_FIXED_DEV_POINT', track, role)
        reference = None
        for arm, step, role in roles:
            folder = root / aliases.get(role, role)
            run = json.loads((folder / 'run.json').read_text())
            completion = json.loads((folder / 'DONE.json').read_text())
            assert run['split'] == 'dev' and run['task'] == track
            assert run['protocol_sha256'] == sha(parent / 'protocol.json')
            expected = MODEL if arm == 'base' else CKPT / f'{track}_{arm}_seed42_v1'
            if arm != 'base' and step == mid:
                expected = expected / f'checkpoint-{mid}'
                assert json.loads((expected / 'trainer_state.json').read_text())['global_step'] == mid
            assert Path(run['model']).resolve() == expected.resolve()
            assert completion['predictions_sha256'] == sha(folder / 'predictions.jsonl')
            predictions = rows(folder / 'predictions.jsonl')
            current = {(p['lang'], p['id']): oh([p['prompt_token_sha256'], p['answer'], p.get('gold_numeric')]) for p in predictions}
            assert len(current) == len(predictions)
            if reference is None:
                reference = current
            assert current == reference, ('Changed dev samples/prompts/references', track, role)
            counts = collections.Counter((p['lang'] for p in predictions))
            assert counts == {lang: protocol['token_stats'][lang + ':dev']['n'] for lang in LANGS}
            summary = track in ['xlsum', 'wikilingua']
            if summary:
                assert run['primary_scorer_version'] == 'v2_explicit_multilingual_sentence_tokenizer'
                assert run['score_correction_protocol_sha256'] == sha(Path(runtime_path('configs', '')) / 'SUMMARY_SCORING_CORRECTION_v2.json')
            scores = json.loads((folder / 'scores.json').read_text())
            provenance[track + ':' + role] = {name: sha(folder / name) for name in ['run.json', 'DONE.json', 'predictions.jsonl', 'scores.json']}
            for lang in LANGS:
                value = scores[lang + ':' + track]
                assert value['samples'] == counts[lang]
                data.append({'track': track, 'language': lang, 'arm': arm, 'segment_step': step, 'logical_cumulative_step': step + final if arm == 'seq_es_th' else step, 'score': value['score'] * (1 if summary else 100), 'metric': 'ROUGE-Lsum F1 points' if summary else 'Tree EM percent' if track == 'mtop' else 'ACC percent', 'samples': value['samples']})
    return (data, provenance)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve prior curve snapshots'
    data, provenance = collect()
    assert len(data) == 7 * 7 * 4
    args.output.mkdir(parents=True)
    figure, axes = plt.subplots(7, 4, figsize=(16, 20), squeeze=False)
    for row, track in enumerate(TRACKS):
        points = [r for r in data if r['track'] == track]
        final = 40 if track == 'sib200' else 200
        mid = final // 2
        summary = track in ['xlsum', 'wikilingua']
        upper = max(20, (int(max((p['score'] for p in points)) / 5) + 2) * 5) if summary else 100
        for column, lang in enumerate(LANGS):
            lookup = {(p['arm'], p['segment_step']): p['score'] for p in points if p['language'] == lang}
            base = lookup['base', 0]
            sequential = [base, lookup['single_es', mid], lookup['single_es', final], lookup['seq_es_th', mid], lookup['seq_es_th', final]]
            direct = [base, lookup['single_th', mid], lookup['single_th', final]]
            axis = axes[row, column]
            axis.plot([0, mid, final, final + mid, 2 * final], sequential, 'o-', color='#245b89', label='Base -> ES -> TH', markersize=4)
            axis.plot([0, mid, final], direct, 's--', color='#c56a1b', label='Base -> TH', markersize=4)
            axis.axvline(final, color='#888888', linestyle=':', linewidth=1)
            axis.set_xlim(-final * 0.03, final * 2.03)
            axis.set_ylim(0, upper)
            axis.set_xticks([0, mid, final, final + mid, 2 * final])
            axis.set_title(track + ' / ' + lang.upper(), fontsize=10)
            axis.set_xlabel('Cumulative optimizer steps', fontsize=8)
            if column == 0:
                axis.set_ylabel('Dev ROUGE-Lsum' if summary else 'Dev Tree EM (%)' if track == 'mtop' else 'Dev ACC (%)', fontsize=9)
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle('Seven independent ES -> TH tracks: fixed development curves', fontsize=14)
    figure.text(0.5, 0.012, 'Same frozen dev IDs, prompt-token hashes and references verified at every point; seed42; no test or checkpoint selection.\nMathOctopus uses held-out train-source math dev, not MGSM/MSVAMP test. Summary units and axis ranges remain separate.\nDots are descriptive fixed-model scores, without training-seed error bars. The dotted vertical line marks the ES/TH boundary.', ha='center', fontsize=9)
    figure.tight_layout(rect=(0, 0.055, 1, 0.97))
    for suffix in ['png', 'pdf']:
        figure.savefig(args.output / ('learning_curves.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(figure)
    with (args.output / 'learning_curves.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    dump(args.output / 'verification.json', {'status': 'ALL_SEVEN_FIXED_DEV_CURVES_RENDERED', 'curve_points': len(data), 'same_dev_ids_prompt_tokens_and_references_verified': True, 'formal_test_included': False, 'checkpoint_selection': False, 'training_seed': 42, 'source_sha256': provenance})
    print(json.dumps({'status': 'ALL_SEVEN_FIXED_DEV_CURVES_RENDERED', 'curve_points': len(data), 'output': str(args.output)}))
if __name__ == '__main__':
    main()
