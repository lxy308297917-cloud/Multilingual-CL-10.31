"""Run one frozen Single/SEQ segment with upstream LF and verify its Final."""
from runtime import runtime_path
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import yaml
from common import ROOT, MODEL, sha, dump
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
    gate = json.loads((Path(runtime_path('configs', '')) / 'P0_GATE.json').read_text())
    assert gate['status'] == 'ALL_SEVEN_AUDITED'
    data = Path(cfg['dataset_dir'])
    assert sha(data / 'protocol.json') == identity['protocol_sha256']
    dev_run = json.loads((a.base_dev / 'run.json').read_text())
    assert dev_run['protocol_sha256'] == identity['protocol_sha256']
    assert (a.base_dev / 'DONE.json').exists()
    engineering = Path(runtime_path('checkpoints', 'mathoctopus_single_es_seed42_v1_engineering_micro2/VALIDATED.json'))
    assert engineering.exists()
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
    py = runtime_path('train_python', '')
    subprocess.run([py, str(ROOT / 'scripts/verify_model_lock.py'), str(ROOT / 'configs/model_17b.lock.json'), str(MODEL)], check=True)
    src = Path(cfg['model_name_or_path'])
    proof = {'source_weights': {p.name: sha(p) for p in src.glob('*.safetensors')}, 'config': str(a.config), 'config_sha256': sha(a.config), 'P0_gate_sha256': sha(Path(runtime_path('configs', '')) / 'P0_GATE.json'), 'base_dev_run_sha256': sha(a.base_dev / 'run.json'), 'optimizer_reset': True, 'scheduler_reset': True}
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
