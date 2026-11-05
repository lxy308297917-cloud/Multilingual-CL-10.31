"""Run the predeclared disjoint TH cohort, preserving the main FFT recipe.

Requires all seven main fixed-Final matrices and the original dev proof;
never replaces old datasets, configurations, models or scores.
"""
from runtime import runtime_path
import argparse, json, os, subprocess, sys, time, shutil
from pathlib import Path
import yaml
from common import ROOT, DATA, MODEL, sha, dump
from validate import validate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--base-dev', type=Path, required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    identity = json.loads(a.config.with_suffix('.identity.json').read_text())
    out = Path(cfg['output_dir'])
    assert not out.exists(), out
    assert sha(a.config) == identity['config_sha256']
    analysis_plan = Path(runtime_path('configs', '')) / 'P3_MTOP_SUPPLEMENT_v1.json'
    assert sha(analysis_plan) == identity['P3_analysis_plan_sha256']
    gate = json.loads((Path(runtime_path('configs', '')) / 'P0_GATE.json').read_text())
    assert gate['status'] == 'ALL_SEVEN_AUDITED'
    main_gate = Path(runtime_path('configs', '')) / 'MAIN_FIXED_FINALS_VALIDATED.json'
    assert json.loads(main_gate.read_text())['status'] == 'ALL_SEVEN_NATIVE_AND_OLD_ABILITY_MATRICES_VERIFIED'
    curves = Path(runtime_path('configs', '')) / 'remaining_tracks_fixed_dev_curves_v1.state.json'
    curve_state = json.loads(curves.read_text())
    assert curve_state['plan_sha256'] == sha(curves.with_name('remaining_tracks_fixed_dev_curves_v1.json'))
    assert curve_state['tasks'] and all((v['status'] == 'DONE' for v in curve_state['tasks'].values())), 'Finish existing main-track evaluation before P3'
    assert shutil.disk_usage(out.parent.resolve()).free >= 60 * 1024 ** 3, 'Insufficient headroom for a new complete training segment'
    data = Path(cfg['dataset_dir'])
    assert sha(data / 'protocol.json') == identity['protocol_sha256']
    dev_run = json.loads((a.base_dev / 'run.json').read_text())
    derived = json.loads((data / 'protocol.json').read_text())
    parent = DATA / 'prepared-v2' / identity['track']
    assert identity['track'] == 'mtop'
    assert derived['parent_protocol_sha256'] == sha(parent / 'protocol.json') == dev_run['protocol_sha256']
    original = json.loads((parent / 'protocol.json').read_text())
    assert derived['training_ids'] == original['disjoint_reserve_ids']
    assert len(derived['training_ids']) == 3200 and (not set(derived['training_ids']) & set(original['training_ids']))
    for rel, h in derived['reference_eval_files_sha256'].items():
        assert sha(parent / rel) == h
    reference = Path(runtime_path('configs', '')) / f"{identity['track']}/{identity['track']}_{identity['arm']}_seed42_v1.yaml"
    reference_cfg = yaml.safe_load(reference.read_text())
    allowed = {'dataset_dir', 'tokenized_path', 'output_dir', 'logging_dir'}
    assert {k: v for k, v in cfg.items() if k not in allowed} == {k: v for k, v in reference_cfg.items() if k not in allowed}, 'Disjoint arm must preserve source model and all optimization settings'
    assert identity['parent_config_sha256'] == sha(reference)
    assert (a.base_dev / 'DONE.json').exists()
    engineering = Path(runtime_path('checkpoints', 'mathoctopus_single_es_seed42_v1_engineering_micro2/VALIDATED.json'))
    assert engineering.exists()
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
    py = runtime_path('train_python', '')
    subprocess.run([py, str(ROOT / 'scripts/verify_model_lock.py'), str(ROOT / 'configs/model_17b.lock.json'), str(MODEL)], check=True)
    src = Path(cfg['model_name_or_path'])
    proof = {'source_weights': {p.name: sha(p) for p in src.glob('*.safetensors')}, 'config': str(a.config), 'config_sha256': sha(a.config), 'P0_gate_sha256': sha(Path(runtime_path('configs', '')) / 'P0_GATE.json'), 'base_dev_run_sha256': sha(a.base_dev / 'run.json'), 'optimizer_reset': True, 'scheduler_reset': True, 'P3_analysis_plan_sha256': sha(analysis_plan), 'main_fixed_final_gate_sha256': sha(main_gate), 'derived_training_protocol_sha256': sha(data / 'protocol.json'), 'unchanged_native_evaluation_protocol_sha256': sha(parent / 'protocol.json')}
    assert proof['source_weights']
    if identity['arm'] == 'seq_es_th':
        parent = json.loads((src / 'VALIDATED.json').read_text())
        assert parent['full_epoch_exact_multiset'] and parent['steps'] == cfg['max_steps']
        assert parent['weights'] == proof['source_weights']
    for rel, h in identity['tokenized_files_sha256'].items():
        assert sha(Path(cfg['tokenized_path']) / rel) == h
    receipt = a.config.with_suffix('.launch_identity.json')
    assert not receipt.exists()
    dump(receipt, proof)
    start = time.monotonic()
    subprocess.run([py, '-m', 'torch.distributed.run', '--nproc_per_node=2', '--standalone', '--nnodes=1', str(ROOT / 'scripts/sft_v1/train_formal.py'), '--config', str(a.config)], check=True)
    result = validate(a.config)
    result['wall_seconds_including_save_and_validation'] = time.monotonic() - start
    result['initialization'] = proof
    dump(out / 'TRAINING_DONE.json', result)
if __name__ == '__main__':
    main()
