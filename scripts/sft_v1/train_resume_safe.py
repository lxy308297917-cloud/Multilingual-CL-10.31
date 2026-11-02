"""Official LF SFT with resume-safe CPUAdam scheduler wiring and applied-LR trace.

Fixes only the optimizer reference used by the existing HF scheduler after
checkpoint loading; no Trainer, optimizer update, or schedule is rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import math
import random
import numpy as np
import torch
from safetensors import safe_open
import yaml
from transformers import TrainerCallback
from train_formal import ExposureAccounting

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from ssu_mask_adapter import apply_gradient_masks


class OptimizerBinding(TrainerCallback):
    def __init__(self, resume):
        self.resume = resume

    def on_train_begin(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
        actual = optimizer
        while hasattr(actual, 'optimizer'):
            actual = actual.optimizer
        scheduler = lr_scheduler
        while hasattr(scheduler, 'scheduler'):
            scheduler = scheduler.scheduler
        assert type(actual).__name__ == 'DeepSpeedCPUAdam'
        assert hasattr(scheduler, 'optimizer') and hasattr(scheduler, 'get_last_lr')
        old = scheduler.optimizer
        assert len(actual.param_groups) == len(old.param_groups)
        before = all(a is b for a, b in zip(actual.param_groups, old.param_groups))
        if self.resume:
            assert scheduler.last_epoch == state.global_step, "Restored scheduler/global-step mismatch"
            scheduler.optimizer = actual
            # The restored scheduler state describes the next update's LR.
            for group, lr in zip(actual.param_groups, scheduler.get_last_lr()):
                group['lr'] = lr
        self.actual, self.scheduler = actual, scheduler
        self.output = Path(args.output_dir)
        record = {'resume': str(self.resume) if self.resume else None,
                  'actual_optimizer_type': type(actual).__name__,
                  'original_scheduler_optimizer_type': type(old).__name__,
                  'shared_group_dictionaries_before': before,
                  'shared_group_dictionaries_after': all(a is b for a,b in zip(actual.param_groups,scheduler.optimizer.param_groups)),
                  'actual_weight_decays': [g['weight_decay'] for g in actual.param_groups],
                  'note': 'CPUAdam constructor logs defaults; actual per-group values govern updates'}
        assert all(g['weight_decay'] == 0 for g in actual.param_groups)
        (self.output/'optimizer-binding.json').write_text(json.dumps(record,indent=2)+'\n')
        self.first_step = True
        if self.resume:
            assert args.world_size == 1, 'Direct state probe is scoped to this single-GPU ZeRO2 path'
            source = Path(self.resume)
            saved = torch.load(next(source.rglob('*optim_states.pt')), map_location='cpu', weights_only=False, mmap=True)['optimizer_state_dict']
            base = saved['base_optimizer_state']
            source_scheduler = torch.load(source/'scheduler.pt',map_location='cpu',weights_only=False)
            assert source_scheduler['last_epoch'] == state.global_step
            assert len(base['param_groups']) == len(source_scheduler['_last_lr'])
            assert all(math.isclose(g['lr'],lr,rel_tol=1e-12,abs_tol=1e-15) for g,lr in zip(base['param_groups'],source_scheduler['_last_lr'])), 'Source checkpoint already has detached applied LR; choose an earlier valid checkpoint'
            tensor_count = 0
            for i, group in enumerate(actual.param_groups):
                assert len(group['params']) == 1
                param = group['params'][0]
                master = saved['single_partition_of_fp32_groups'][i]
                assert torch.equal(param.detach().reshape(-1)[:master.numel()], master.reshape(-1)), 'FP32 master not restored exactly'
                tensor_count += 1
                reference = base['state'][base['param_groups'][i]['params'][0]]
                restored = actual.state[param]
                assert set(reference) == set(restored)
                for key, value in reference.items():
                    if isinstance(value, torch.Tensor):
                        assert torch.equal(value, restored[key]), 'Adam tensor restore mismatch: '+key
                        tensor_count += 1
                    else:
                        assert value == restored[key], 'Adam scalar restore mismatch: '+key
            physical = model
            while hasattr(physical, 'module'):
                physical = physical.module
            model_count = 0
            with safe_open(source/'model.safetensors',framework='pt',device='cpu') as stream:
                for name, param in physical.named_parameters():
                    assert name in stream.keys(), name
                    assert torch.equal(param.detach().cpu(), stream.get_tensor(name)), 'BF16 model not restored exactly: '+name
                    model_count += 1
            rng_paths = list(source.glob('rng_state*.pth'))
            assert len(rng_paths) == 1
            self.saved_rng = torch.load(rng_paths[0],map_location='cpu',weights_only=False)
            self.restore_record = {'status':'validated model and optimizer before first update',
                'source_checkpoint':str(source.resolve()),'global_step':state.global_step,
                'unique_model_parameters_compared':model_count,'fp32_master_and_adam_tensors_compared':tensor_count,
                'all_model_weights_exact':True,'all_fp32_master_and_adam_tensors_exact':True,
                'adam_scalar_states_exact':True,'source_optimizer_schedule_consistent':True,'rng_checked':False}
            (self.output/'restored-state-integrity.json').write_text(json.dumps(self.restore_record,indent=2)+'\n')
            del saved,base


    def on_step_begin(self, args, state, control, **kwargs):
        if self.resume and self.first_step:
            saved = self.saved_rng
            current_np = np.random.get_state()
            numpy_equal = current_np[0] == saved['numpy'][0] and np.array_equal(current_np[1],saved['numpy'][1]) and current_np[2:] == saved['numpy'][2:]
            checks = {'python':random.getstate() == saved['python'],
                      'numpy':bool(numpy_equal), 'torch_cpu':torch.equal(torch.random.get_rng_state(),saved['cpu'])}
            if 'cuda' in saved:
                expected = saved['cuda'] if isinstance(saved['cuda'],list) else [saved['cuda']]
                current = torch.cuda.random.get_rng_state_all()
                checks['cuda'] = len(expected)==len(current) and all(torch.equal(a,b) for a,b in zip(expected,current))
            self.restore_record['rng_checks'] = checks
            self.restore_record['rng_checked'] = True
            (self.output/'restored-state-integrity.json').write_text(json.dumps(self.restore_record,indent=2)+'\n')
            assert all(checks.values()), 'RNG restore mismatch at first resumed update'
            self.first_step = False
        actual = [float(g['lr']) for g in self.actual.param_groups]
        reported = [float(v) for v in self.scheduler.get_last_lr()]
        assert len(actual) == len(reported)
        assert all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15) for a,b in zip(actual,reported)), 'Applied optimizer LR detached from restored scheduler'
        with (self.output/f'applied-lr-rank{args.process_index}.jsonl').open('a') as stream:
            stream.write(json.dumps({'update_step':state.global_step+1,'actual_lrs':actual,'scheduler_last_lrs':reported})+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--stop-at',type=int)
    p.add_argument('--ssu',action='store_true')
    a = p.parse_args()
    config = yaml.safe_load(a.config.read_text())
    resume = a.resume or config.get('resume_from_checkpoint')
    if resume:config['resume_from_checkpoint']=str(resume)
    assert config['stage']=='sft' and config['finetuning_type']=='full'
    ds=json.loads(Path(config['deepspeed']).read_text())
    assert ds['zero_optimization']['stage']==2
    assert ds['zero_optimization']['offload_optimizer']['device']=='cpu'
    assert config['weight_decay']==0
    if a.ssu:
        import llamafactory.train.sft.workflow as workflow
        lock=json.loads((ROOT/'configs/mask_17b.lock.json').read_text())
        mask=Path(lock['mask_path']);assert hashlib.sha256(mask.read_bytes()).hexdigest()==lock['mask_sha256']
        original=workflow.load_model
        def load_masked(tokenizer,model_args,finetuning_args,is_trainable=False):
            model=original(tokenizer,model_args,finetuning_args,is_trainable)
            report=apply_gradient_masks(model,mask)
            assert report['total_unique_elements']==lock['total_unique_elements']
            out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
            (out/'ssu-mask-applied.json').write_text(json.dumps({'mask_sha256':lock['mask_sha256'],'report':report},indent=2)+'\n')
            return model
        workflow.load_model=load_masked
    from llamafactory.train.tuner import run_exp
    run_exp(config,callbacks=[ExposureAccounting(resume,a.stop_at),OptimizerBinding(resume)])


if __name__=='__main__':main()
