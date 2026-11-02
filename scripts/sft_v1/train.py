"""Official LLaMA-Factory run_exp plus accounting/engineering callbacks only."""
import argparse
import json
import os
from pathlib import Path
import time
import resource

import torch
import yaml
from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp


class Accounting(TrainerCallback):
    def __init__(self, resume=None, stop_at=None):
        self.resume=resume; self.stop_at=stop_at; self.counts={'samples':0,'input_tokens':0,'shifted_assistant_tokens':0,'padded_positions':0}
        self.started=time.monotonic(); self.hook=None

    def on_train_begin(self,args,state,control,model=None,**kwargs):
        if self.resume:
            p=Path(self.resume)/f'accounting-rank{args.process_index}.json'
            assert p.exists(), 'Resume must retain prior exact token counters'
            old=json.loads(p.read_text()); self.counts=old['local_counts']
            assert old['step']==state.global_step, (old['step'],state.global_step)
        self.initial_step=state.global_step
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        total=sum(p.numel() for p in model.parameters())
        self.parameters={'total_local_parameters':total,'trainable_local_parameters':trainable}
        def observe(module,inputs,kw):
            if not module.training:return
            ids=kw.get('input_ids'); labels=kw.get('labels'); mask=kw.get('attention_mask')
            assert ids is not None and labels is not None
            assert (labels==-100).any() and (labels[:,1:]!=-100).any(), 'SFT supervision mask invalid'
            self.counts['samples']+=ids.shape[0]
            self.counts['input_tokens']+=int(mask.sum()) if mask is not None else ids.numel()
            self.counts['shifted_assistant_tokens']+=int((labels[:,1:]!=-100).sum())
            self.counts['padded_positions']+=ids.numel()
        self.hook=model.register_forward_pre_hook(observe,with_kwargs=True)

    def record(self,args,state,path):
        values=torch.tensor(list(self.counts.values()),dtype=torch.int64,device=args.device)
        if torch.distributed.is_initialized():torch.distributed.all_reduce(values)
        result={'step':state.global_step,'resume_initial_step':self.initial_step,'local_counts':self.counts.copy(),'global_counts':dict(zip(self.counts,map(int,values.tolist()))),'elapsed_this_process_seconds':time.monotonic()-self.started,'max_memory_allocated':torch.cuda.max_memory_allocated(),'max_memory_reserved':torch.cuda.max_memory_reserved(),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'parameters':self.parameters,'world_size':args.world_size,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'log_history':state.log_history}
        path.mkdir(parents=True,exist_ok=True)
        (path/f'accounting-rank{args.process_index}.json').write_text(json.dumps(result,indent=2)+'\n')

    def on_step_end(self,args,state,control,**kwargs):
        if self.stop_at and state.global_step>=self.stop_at:
            control.should_save=True;control.should_training_stop=True
        return control

    def on_log(self,args,state,control,logs=None,**kwargs):
        if logs and 'loss' in logs:
            import math
            if not math.isfinite(logs['loss']):raise RuntimeError('Nonfinite training loss')

    def on_save(self,args,state,control,**kwargs):
        self.record(args,state,Path(args.output_dir)/f'checkpoint-{state.global_step}')

    def on_train_end(self,args,state,control,**kwargs):
        self.record(args,state,Path(args.output_dir))
        if self.hook:self.hook.remove()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--stop-at',type=int);ap.add_argument('--resume',type=Path);a=ap.parse_args()
    config=yaml.safe_load(a.config.read_text())
    if a.resume:config['resume_from_checkpoint']=str(a.resume)
    run_exp(config,callbacks=[Accounting(a.resume,a.stop_at)])


if __name__=='__main__':main()
