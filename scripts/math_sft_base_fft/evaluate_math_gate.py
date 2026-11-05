"""Evaluate only the predeclared math gate while full general tests continue.

This is not whole-experiment completion and never launches or stops jobs. The
complete final analysis must reproduce the exact same gate details.
"""
from profile_paths import resolve
import json
from pathlib import Path
from common import *
from aggregate_route_a import paired, math_gate_passes

def main():
    result = ROOT / 'results/math_sft_base_fft'
    selection = json.loads((result / 'base-language-screening-v1/selection.json').read_text())
    langs = selection['selected_languages']
    missing = [str(Path('A') / l / f'epoch{e}' / 'math/GENERATION_DONE.json') for l in langs for e in [1, 2] if not (result / 'A' / l / f'epoch{e}' / 'math/GENERATION_DONE.json').exists()]
    if missing:
        print(json.dumps({'status': 'NOT_RUN', 'scope': 'Frozen math gate only', 'waiting_for': missing}))
        return
    out = result / 'routeA-math-gate-v1'
    assert not out.exists()
    base = [r for i in [1, 2, 3] for r in rows(result / f'base-census-worker{i}-v1/predictions.jsonl')]
    gates = []
    sources = {}
    reference = json.loads((result / 'base-census-worker1-v1/run.json').read_text())
    assets = {}
    for i in [1, 2, 3]:
        manifest = json.loads((result / f'base-census-worker{i}-v1/run.json').read_text())
        assert manifest['protocol_sha256'] == reference['protocol_sha256'] and manifest['generation'] == reference['generation'] and (manifest['weights_sha256'] == reference['weights_sha256'])
        assets.update(manifest['assets'])
    for lang in langs:
        for ep in [1, 2]:
            folder = result / 'A' / lang / f'epoch{ep}' / 'math'
            data = rows(folder / 'predictions.jsonl')
            cs = []
            run = json.loads((folder / 'run.json').read_text())
            training = json.loads((Path(resolve('CHECKPOINTS')) / f'mathoctopus_{lang}_full_2epochs_seed42_v1/TRAINING_DONE.json').read_text())
            assert run['weights_sha256'] == training['epochs'][ep - 1]['weights_sha256']
            assert run['protocol_sha256'] == reference['protocol_sha256'] and run['generation'] == reference['generation'] and (run['assets'] == {lang + '_eval.jsonl': assets[lang + '_eval.jsonl']})
            for task, n in [('mgsm_rev2', 250), ('msvamp', 1000)]:
                aa = {r['id']: r for r in base if r['lang'] == lang and r['benchmark'] == task}
                bb = {r['id']: r for r in data if r['benchmark'] == task}
                assert len(aa) == len(bb) == n and aa.keys() == bb.keys()
                ids = sorted(aa)
                assert all((aa[k]['prompt_token_sha256'] == bb[k]['prompt_token_sha256'] for k in ids))
                cs.append(dict(language=lang, before_epoch=0, after_epoch=ep, task=task, **paired([aa[k]['correct'] for k in ids], [bb[k]['correct'] for k in ids])))
            gates.append({'language': lang, 'epoch': ep, 'passes': math_gate_passes(cs), 'metrics': cs})
            sources[f'{lang}-E{ep}'] = {'predictions_sha256': sha(folder / 'predictions.jsonl'), 'run_manifest_sha256': sha(folder / 'run.json')}
    counts = {str(ep): sum((x['passes'] for x in gates if x['epoch'] == ep)) for ep in [1, 2]}
    success = any((v >= 2 for v in counts.values()))
    decision = {'route': 'A', 'status': 'PASS' if success else 'FAIL_FIXED_E1_AND_E2_GATE', 'passing_languages_per_same_epoch': counts, 'gate_details': gates, 'route_B': 'NOT_RUN_CONDITION_NOT_MET' if success else 'AUTHORIZED_CONDITIONAL_READY_FOR_RESOURCE_AND_DATA_AUDIT', 'EN_decline_used_as_failure': False, 'all_epochs_and_languages_retained': True, 'scope': 'BENCHMARK_GUIDED_EXPLORATORY_SINGLE_SEED'}
    out.mkdir()
    dump(out / 'decision.json', decision)
    dump(out / 'source_identity.json', sources)
    dump(out / 'DONE.json', {'status': 'COMPLETE', 'scope': 'Frozen target math gate only. General evaluations must continue to full completion; no job is cancelled and the thread goal is NOT complete.', 'gate_function_source_sha256': sha(Path(__file__).with_name('aggregate_route_a.py')), 'B_requires_resource_and_data_audit_before_training': not success})
    print(json.dumps({'status': 'COMPLETE', 'route_A': decision['status'], 'passing_languages_per_same_epoch': counts, 'route_B': decision['route_B']}))
if __name__ == '__main__':
    main()
