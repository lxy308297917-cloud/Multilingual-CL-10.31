"""Official LLaMA-Factory SFT with existing exposure accounting and optional SSU hooks."""
from profile_paths import resolve
import sys, argparse, json, hashlib, shutil, datetime
from pathlib import Path
import yaml, torch
import importlib.util
from deepspeed.ops.adam import cpu_adam
R = Path(resolve('ROOT'))
sys.path[:0] = [str(R / 'scripts/sft_v1'), str(R / 'scripts')]
from train_formal import ExposureAccounting
from train_resume_safe import OptimizerBinding
from llamafactory.train.tuner import run_exp

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(x)
    return h.hexdigest()

class BudgetAccounting(ExposureAccounting):

    def on_train_begin(self, args, state, control, **kw):
        super().on_train_begin(args, state, control, **kw)
        assert args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps == 16
        assert state.max_steps == self.expected_steps, (state.max_steps, self.expected_steps)

    def on_train_end(self, args, state, control, **kw):
        super().on_train_end(args, state, control, **kw)
        assert state.global_step == self.expected_steps
        assert self.counts['samples'] == self.expected_steps * 16

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--mask', type=Path)
    a = p.parse_args()
    config = yaml.safe_load(a.config.read_text())
    out = Path(config['output_dir'])
    assert not out.exists(), 'No overwrites or implicit resumes'
    lock = json.loads((R / 'configs/aya_4b_20261105/data.lock.json').read_text())
    base = Path(config['model_name_or_path'])
    assert str(base) == resolve('MODEL')
    assert lock['rows'] == 6400
    for rel, digest in lock['files'].items():
        assert sha(Path(config['dataset_dir']) / rel) == digest, rel
    if a.mask:
        from ssu_mask_adapter import apply_gradient_masks
        import llamafactory.train.sft.workflow as workflow
        assert config['finetuning_type'] == 'full' and config['weight_decay'] == 0
        assert json.loads(Path(config['deepspeed']).read_text())['zero_optimization']['stage'] == 2
        original = workflow.load_model

        def load(*args, **kwargs):
            model = original(*args, **kwargs)
            report = apply_gradient_masks(model, a.mask)
            out.mkdir(parents=True, exist_ok=True)
            (out / 'mask-applied.json').write_text(json.dumps({'mask_sha256': sha(a.mask), 'report': report, 'limitation': 'Same English Tulu calibration selection as old SSU; known mathematics bias'}, indent=2) + '\n')
            return model
        workflow.load_model = load
    library = Path(resolve('CPU_ADAM_SO'))
    spec = importlib.util.spec_from_file_location('cpu_adam', library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cpu_adam.CPUAdamBuilder.load = lambda self: module
    cb = BudgetAccounting()
    cb.expected_steps = int(config.get('max_steps', 400))
    started = datetime.datetime.now().astimezone().isoformat()
    run_exp(config, callbacks=[cb, OptimizerBinding(None)])
    shutil.copyfile(base / 'generation_config.json', out / 'generation_config.json')
    state = json.loads((out / 'trainer_state.json').read_text())
    assert state['global_step'] == cb.expected_steps
    accounting = json.loads((out / 'accounting-rank0.json').read_text())
    assert accounting['global_counts']['samples'] == cb.expected_steps * 16
    weights = {p.name: sha(p) for p in out.glob('*.safetensors')}
    assert weights
    result = {'status': 'SMOKE_DONE' if cb.expected_steps != 400 else 'TRAINING_DONE', 'steps': state['global_step'], 'samples': accounting['global_counts']['samples'], 'exposure': accounting['global_counts'], 'config_sha256': sha(a.config), 'data_manifest_sha256': sha(R / 'configs/aya_4b_20261105/data.lock.json'), 'model_lock_sha256': sha(R / 'configs/aya_4b_20261105/model.lock.json'), 'weights_sha256': weights, 'started_at': started, 'ended_at': datetime.datetime.now().astimezone().isoformat(), 'saved_optimizer_state': False, 'final_checkpoint_policy': 'Fixed budget Final, no benchmark selection or English early stop'}
    (out / ('SMOKE_DONE.json' if cb.expected_steps != 400 else 'TRAINING_DONE.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    main()
