"""Fixed dev learning curves. Midpoints never select formal Finals."""
from runtime import runtime_path
import os, sys, json, csv, collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, runtime_path('project', 'scripts/unified_language_cl'))
from common import ROOT, rows, sha, dump
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LANGS = ['es', 'th', 'fr', 'en']
data = []
receipts = {}
for track in ['massive', 'mathoctopus']:
    root = Path(runtime_path('results', '')) / '' / track
    aliases = json.loads((root / 'run_aliases.json').read_text())
    sources = {}
    for role in ['base-dev-v1'] + [a + '-' + s for a in ['single_es', 'single_th', 'seq_es_th'] for s in ['dev100-v1', 'dev-v1']]:
        p = root / aliases.get(role, role)
        assert (p / 'DONE.json').exists(), p
        sources[role] = rows(p / 'predictions.jsonl')
        receipts[track + ':' + role] = {'prediction_sha256': sha(p / 'predictions.jsonl'), 'scores_sha256': sha(p / 'scores.json')}
    reference = {(r['lang'], r['id']): (r['prompt_token_sha256'], r['answer']) for r in sources['base-dev-v1']}
    for role, rs in sources.items():
        assert {(r['lang'], r['id']): (r['prompt_token_sha256'], r['answer']) for r in rs} == reference
        scores = json.loads((root / aliases.get(role, role) / 'scores.json').read_text())
        for lang in LANGS:
            value = scores[lang + ':' + track]['score'] * 100
            arm = 'base' if role.startswith('base') else role.split('-dev')[0]
            step = 0 if arm == 'base' else 100 if 'dev100' in role else 200
            data.append({'track': track, 'lang': lang, 'arm': arm, 'segment_step': step, 'acc_percent': value, 'samples': scores[lang + ':' + track]['samples']})
fig, axs = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
for ti, t in enumerate(['massive', 'mathoctopus']):
    for li, l in enumerate(LANGS):
        lookup = {(r['arm'], r['segment_step']): r['acc_percent'] for r in data if r['track'] == t and r['lang'] == l}
        b = lookup['base', 0]
        seq = [b, lookup['single_es', 100], lookup['single_es', 200], lookup['seq_es_th', 100], lookup['seq_es_th', 200]]
        direct = [b, lookup['single_th', 100], lookup['single_th', 200]]
        ax = axs[ti, li]
        ax.plot([0, 100, 200, 300, 400], seq, 'o-', label='Base -> ES -> TH')
        ax.plot([0, 100, 200], direct, 's--', label='Base -> TH')
        ax.axvline(200, color='grey', ls=':', alpha=0.6)
        ax.set_ylim(0, 100)
        ax.set_title(t + ' / ' + l.upper())
        ax.grid(alpha=0.2)
        ax.set_xlabel('Optimizer steps')
        ax.set_ylabel('Dev ACC (%)' if li == 0 else '')
axs[0, 0].legend(fontsize=8)
fig.suptitle('Frozen dev curves; seed42; fixed Final budgets, no test selection\nMathOctopus dev is held-out GSM8K train-source data, not formal MGSM/MSVAMP test', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = Path(runtime_path('results', '')) / 'first_batch_dev_curves_v1'
assert not out.exists()
out.mkdir()
fig.savefig(out / 'learning_curves.png', dpi=180)
fig.savefig(out / 'learning_curves.pdf')
with (out / 'learning_curves.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(data[0]))
    w.writeheader()
    w.writerows(data)
dump(out / 'verification.json', {'same_dev_ids_prompts_references_verified': True, 'sources': receipts, 'training_seed': 42, 'used_for_checkpoint_selection': False, 'formal_tests_included': False})
print(json.dumps({'status': 'DONE', 'curve_points': len(data)}))
