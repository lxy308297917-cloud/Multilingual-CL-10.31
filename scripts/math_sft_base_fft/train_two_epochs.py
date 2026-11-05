"""Official LF training; two true epochs, one optimizer/scheduler, no score stop."""
import sys, argparse, json, math, shutil, os
from pathlib import Path
from common import *
sys.path.insert(0, str(ROOT / 'scripts/sft_v1'))
from train_formal import ExposureAccounting
from llamafactory.train.tuner import run_exp
import yaml, torch

class EpochAccounting(ExposureAccounting):

    def on_train_begin(self, args, state, control, **kw):
        super().on_train_begin(args, state, control, **kw)
        assert args.num_train_epochs == 2 and args.learning_rate == 5e-06
        assert args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps == 16
        self.optimizer_id = id(kw['optimizer'])
        self.scheduler_id = id(kw['lr_scheduler'])
        self.expected_steps = math.ceil(self.dataset_samples / 16)
        assert state.max_steps == 2 * self.expected_steps, (state.max_steps, self.expected_steps)

    def on_epoch_end(self, args, state, control, **kw):
        assert id(kw['optimizer']) == self.optimizer_id and id(kw['lr_scheduler']) == self.scheduler_id
        ep = round(state.epoch)
        assert ep in [1, 2] and abs(ep - state.epoch) < 1e-08
        assert state.global_step == ep * self.expected_steps, (ep, state.global_step, self.expected_steps)
        control.should_save = True
        return control

    def on_save(self, args, state, control, **kw):
        super().on_save(args, state, control, **kw)
        path = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        if args.process_index == 0:
            shutil.copyfile(MODEL / 'generation_config.json', path / 'generation_config.json')
            fs = sorted(path.glob('*.safetensors'))
            assert fs
            dump(path / 'EPOCH_CHECKPOINT.json', {'epoch': state.epoch, 'step': state.global_step, 'optimizer_object_identity': self.optimizer_id, 'scheduler_object_identity': self.scheduler_id, 'epoch_boundary_lrs': [float(g['lr']) for g in getattr(kw['optimizer'], 'param_groups', [])], 'expected_steps_per_epoch': self.expected_steps, 'num_train_epochs': 2, 'continuous_optimizer_scheduler': True, 'save_only_model': True, 'optimizer_state_on_disk': False, 'weights_sha256': {p.name: sha(p) for p in fs}, 'training_data_sha256': sha(self.data_manifest), 'config_sha256': sha(self.config_path)})
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_train_end(self, args, state, control, **kw):
        super().on_train_end(args, state, control, **kw)
        assert state.global_step == 2 * self.expected_steps and abs(state.epoch - 2) < 1e-08

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    a = ap.parse_args()
    config = yaml.safe_load(a.config.read_text())
    target = Path(config['output_dir'])
    manifest = Path(config['dataset_dir']) / 'training_data.json'
    data = json.loads(manifest.read_text())
    callback = EpochAccounting()
    callback.dataset_samples = data['samples']
    callback.data_manifest = manifest
    callback.config_path = a.config
    run_exp(config, callbacks=[callback])
if __name__ == '__main__':
    main()
