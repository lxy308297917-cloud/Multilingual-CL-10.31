"""Execution-only NLI micro-batch probe; same effective batch and first64 IDs.

Not a scientific model; no dev/test scoring. Compare steady step duration and
peak allocated VRAM for micro2 vs micro4 before freezing NLI formal configs.
"""
from runtime import runtime_path
import argparse, json, sys, time
from pathlib import Path
import yaml
import torch
from transformers import TrainerCallback
from common import ROOT, dump, sha
sys.path.insert(0, str(ROOT / 'scripts/sft_v1'))
from train_formal import ExposureAccounting
from llamafactory.train.tuner import run_exp

class Timing(TrainerCallback):

    def __init__(self):
        self.times = []

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.begin = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.times.append(time.monotonic() - self.begin)

    def on_train_end(self, args, state, control, **kwargs):
        result = {'per_step_seconds': self.times, 'steady_last_two_step_seconds': sum(self.times[-2:]) / len(self.times[-2:]), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'steps': state.global_step, 'scope': 'engineering_only_not_a_scientific_comparison'}
        dump(Path(args.output_dir) / f'timing-rank{args.process_index}.json', result)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    assert cfg['max_steps'] == 200 and cfg['save_only_model'] is True and (cfg['per_device_train_batch_size'] in [2, 4])
    assert not Path(cfg['output_dir']).exists()
    run_exp(cfg, callbacks=[ExposureAccounting(stop_at=4), Timing()])
    if int(__import__('os').environ.get('RANK', '0')) == 0:
        out = Path(cfg['output_dir'])
        state = json.loads((out / 'trainer_state.json').read_text())
        assert state['global_step'] == 4
        accounts = [json.loads((out / f'accounting-rank{r}.json').read_text()) for r in [0, 1]]
        assert all((x['global_counts']['samples'] == 64 for x in accounts))
        dump(out / 'ENGINEERING_DONE.json', {'scope': 'ENGINEERING_ONLY', 'global_counts': accounts[0]['global_counts'], 'config_sha256': sha(a.config), 'per_rank_timing': [json.loads((out / f'timing-rank{r}.json').read_text()) for r in [0, 1]]})
if __name__ == '__main__':
    main()
