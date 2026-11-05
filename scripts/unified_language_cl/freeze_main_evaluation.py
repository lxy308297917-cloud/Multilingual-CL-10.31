"""Derive P3 prerequisites from completed reproduction evidence, never old markers.

No generation, training or scheduling occurs. Derived DONE entries attest to
verified outputs; they are not scheduler histories or fabricated process exits.
"""
import collections
import json
from pathlib import Path
from common import ROOT, DATA, MODEL, sha, rows, dump
from runtime import runtime_path
from make_configs import CKPT
from render_stage_effects import load_all, TRACKS
from aggregate_retention import directory as retention_directory, read as read_retention


WEIGHTS = {}


def check_run(track, split, folder, model, protocol):
    assert (folder/'DONE.json').exists(), ('Missing completed evaluation', folder)
    run = json.loads((folder/'run.json').read_text())
    done = json.loads((folder/'DONE.json').read_text())
    parent = DATA/('prepared-v2' if track == 'mtop' else 'prepared-v1')/track
    assert run['protocol_sha256'] == sha(parent/'protocol.json')
    assert Path(run['model']).resolve() == model.resolve()
    assert run['task'] == track and run['split'] == split
    assert set(run['langs']) == {'es', 'th', 'fr', 'en'}
    expected_assets = {f'{lang}_{split}.jsonl': protocol['files'][f'{lang}_{split}.jsonl'] for lang in run['langs']}
    assert run['assets'] == expected_assets
    for name, digest in expected_assets.items():
        assert sha(parent/name) == digest
    model_key = str(model.resolve())
    if model_key not in WEIGHTS:
        WEIGHTS[model_key] = {p.name:sha(p) for p in model.glob('*.safetensors')}
        assert WEIGHTS[model_key]
    assert run['weights_sha256'] == WEIGHTS[model_key]
    assert done['predictions_sha256'] == sha(folder/'predictions.jsonl')
    predictions = rows(folder/'predictions.jsonl')
    counts = collections.Counter(p['lang'] for p in predictions)
    assert counts == {lang:protocol['token_stats'][lang+':'+split]['n'] for lang in run['langs']}
    assert len({(p['lang'],p['id']) for p in predictions}) == len(predictions)
    if track in ['xlsum', 'wikilingua']:
        assert run['primary_scorer_version'] == 'v2_explicit_multilingual_sentence_tokenizer'
        assert run['score_correction_protocol_sha256'] == sha(Path(runtime_path('configs'))/'SUMMARY_SCORING_CORRECTION_v2.json')
    return {'run_sha256':sha(folder/'run.json'), 'done_sha256':sha(folder/'DONE.json'),
            'predictions_sha256':done['predictions_sha256'], 'samples':len(predictions)}


def main():
    cfg = Path(runtime_path('configs'))
    marker = cfg/'MAIN_FIXED_FINALS_VALIDATED.json'
    plan_file = cfg/'remaining_tracks_fixed_dev_curves_v1.json'
    state_file = cfg/'remaining_tracks_fixed_dev_curves_v1.state.json'
    assert not marker.exists() and not plan_file.exists() and not state_file.exists(), 'Use a fresh reproduction namespace'
    # Fail before any expensive data reads if the seven complete matrices do
    # not exist; this never approves a partial or test-selected model set.
    native, general, matrices = load_all()
    assert set(native) == set(TRACKS) == set(general)
    checked = {}
    for track in TRACKS:
        parent = DATA/('prepared-v2' if track == 'mtop' else 'prepared-v1')/track
        protocol = json.loads((parent/'protocol.json').read_text())
        root = Path(runtime_path('results'))/track
        aliases = json.loads((root/'run_aliases.json').read_text()) if (root/'run_aliases.json').exists() else {}
        for model in ['base','single_es','single_th','seq_es_th']:
            expected = MODEL if model == 'base' else CKPT/f'{track}_{model}_seed42_v1'
            if model != 'base':
                validation = json.loads((expected/'VALIDATED.json').read_text())
                assert validation['full_epoch_exact_multiset'] and validation['checkpoint_optimizer_preserved']
                assert validation['steps'] == protocol['steps']
            for split in ['dev','test']:
                suffix = model+'-'+split+'-v1'
                if track == 'mtop' and model == 'base' and split == 'dev':
                    suffix = 'base-dev-v2-syntax'
                suffix = aliases.get(suffix,suffix)
                assert Path(suffix).name == suffix
                folder = root/suffix
                proof = check_run(track,split,folder,expected,protocol)
                effect = json.loads((root/f'comparison-{split}-v1/effects.json').read_text())
                assert effect['sources'][model]['run_sha256'] == proof['run_sha256']
                checked[track+':'+model+':'+split] = proof
            retained = retention_directory(track,model)
            assert (retained/'DONE.json').exists()
            receipt = json.loads((retained/'DONE.json').read_text())
            assert receipt['weights_sha256'] == WEIGHTS[str(expected.resolve())]
            measured, hashes = read_retention(retained)
            assert measured and hashes
            checked[track+':'+model+':retention'] = {'done_sha256':sha(retained/'DONE.json'), 'input_result_files':len(hashes)}
    # Match the original predeclared five-track midpoint set. The first two
    # tracks have separate earlier curve reports and are not replaced here.
    tasks, state = [], {}
    for track in ['mtop','xnli','xlsum','sib200','wikilingua']:
        parent = DATA/('prepared-v2' if track == 'mtop' else 'prepared-v1')/track
        protocol = json.loads((parent/'protocol.json').read_text())
        mid = protocol['steps']//2
        root = Path(runtime_path('results'))/track
        for arm in ['single_es','single_th','seq_es_th']:
            model = CKPT/f'{track}_{arm}_seed42_v1'/f'checkpoint-{mid}'
            trainer = json.loads((model/'trainer_state.json').read_text())
            assert trainer['global_step'] == mid
            folder = root/f'{arm}-dev{mid}-v1'
            proof = check_run(track,'dev',folder,model,protocol)
            name = f'curve-{track}-{arm}-dev{mid}-v1'
            tasks.append({'name':name,'scope':'Verified already-completed fixed midpoint dev, not a submitted job','checkpoint_step':mid,'evidence':proof})
            state[name] = {'status':'DONE','basis':'Actual completed-output validation; no scheduler/process exit is inferred','evidence':proof}
    cfg.mkdir(parents=True,exist_ok=True)
    dump(plan_file,{'scope':'Reproduction evidence plan, not a GPU scheduling queue','tasks':tasks,'max_user_gpus':4})
    dump(state_file,{'plan_sha256':sha(plan_file),'tasks':state,'derived_from_real_evaluation_outputs':True})
    dump(marker,{'status':'ALL_SEVEN_NATIVE_AND_OLD_ABILITY_MATRICES_VERIFIED','tracks':TRACKS,
                 'scope':'Independently regenerated reproduction evidence, not copied historical successes',
                 'formal_test_selection':False,'matrix_source_sha256':matrices,
                 'checked_final_and_base_runs':checked,'fixed_midpoint_receipt_sha256':sha(state_file)})
    print(json.dumps({'status':'ALL_SEVEN_NATIVE_AND_OLD_ABILITY_MATRICES_VERIFIED','fixed_midpoint_runs':len(tasks),'new_gpu_work':False}))


if __name__ == '__main__':
    main()
