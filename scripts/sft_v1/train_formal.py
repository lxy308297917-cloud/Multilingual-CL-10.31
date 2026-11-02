"""Official LF SFT with parameter-aware, per-example exposure accounting.

The initial train.py remains frozen for live legacy continuation. This entry
extends accounting only; it does not implement a Trainer or alter optimization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import yaml
import torch
import math
from train import Accounting
from llamafactory.train.tuner import run_exp


class ExposureAccounting(Accounting):
    def __init__(self,resume=None,stop_at=None):
        super().__init__(resume,stop_at);self.exposure=[];self.exposure_hook=None;self.loss_hook=None

    def on_train_begin(self,args,state,control,model=None,**kwargs):
        super().on_train_begin(args,state,control,model=model,**kwargs)
        # ZeRO3 parameters have numel()==0 between gathers; ds_numel is global.
        self.parameters={'unique_model_parameters':sum(getattr(p,'ds_numel',p.numel()) for p in model.parameters()),'trainable_unique_parameters':sum(getattr(p,'ds_numel',p.numel()) for p in model.parameters() if p.requires_grad)}
        self.state_ref=state;self.micro=0
        if self.resume:
            p=Path(self.resume)/f'exposure-rank{args.process_index}.jsonl'
            if p.exists():self.exposure=[json.loads(line) for line in p.open()]
        def observe(module,inputs,kw):
            if not module.training:return
            ids=kw['input_ids'];labels=kw['labels'];mask=kw['attention_mask']
            for j in range(ids.shape[0]):
                keep=mask[j].bool();tokens=ids[j][keep].tolist();targets=labels[j][keep].tolist()
                sha=hashlib.sha256(json.dumps([tokens,targets],separators=(',',':')).encode()).hexdigest()
                self.exposure.append({'update_step':self.state_ref.global_step+1,'microstep':self.micro,'rank':args.process_index,'row_in_microbatch':j,'input_label_sha256':sha,'input_tokens':len(tokens),'shifted_assistant_tokens':sum(t!=-100 for t in targets[1:])})
            self.micro+=1
        self.exposure_hook=model.register_forward_pre_hook(observe,with_kwargs=True)
        def finite_loss(module,inputs,output):
            loss=getattr(output,'loss',None)
            if module.training and loss is not None and not bool(torch.isfinite(loss.detach()).all()):
                raise RuntimeError('Nonfinite actual forward loss before backward')
        self.loss_hook=model.register_forward_hook(finite_loss)

    def on_step_begin(self,args,state,control,**kwargs):
        self.micro=0

    def record(self,args,state,path):
        super().record(args,state,path)
        p=path/f'exposure-rank{args.process_index}.jsonl'
        p.write_text(''.join(json.dumps(r)+'\n' for r in self.exposure))
        f=path/f'accounting-rank{args.process_index}.json';j=json.loads(f.read_text());j['exposure_file_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();j['trace_rows']=len(self.exposure);j['trace_covers_all_local_samples']=len(self.exposure)==self.counts['samples'];f.write_text(json.dumps(j,indent=2)+'\n')

    def on_train_end(self,args,state,control,**kwargs):
        super().on_train_end(args,state,control,**kwargs)
        if self.exposure_hook:self.exposure_hook.remove()
        if self.loss_hook:self.loss_hook.remove()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--stop-at',type=int);ap.add_argument('--resume',type=Path);a=ap.parse_args()
    config=yaml.safe_load(a.config.read_text());resume=a.resume or config.get('resume_from_checkpoint')
    if resume:config['resume_from_checkpoint']=str(resume)
    run_exp(config,callbacks=[ExposureAccounting(resume,a.stop_at)])


if __name__=='__main__':main()
