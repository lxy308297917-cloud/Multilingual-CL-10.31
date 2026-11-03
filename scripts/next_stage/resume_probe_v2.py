"""Four-step engineering validation; official LF/DeepSpeed training unchanged.

Only reconnect the restored scheduler to the actual AdamW, as in the already
validated SST adapter; no gradient or weight projection is performed.
"""
from pathlib import Path
import argparse,json,sys,math
import yaml
import torch
from transformers import TrainerCallback
r=Path(__file__).resolve().parents[2];sys.path.insert(0,str(r/'scripts/sft_v1'))
from train_formal import ExposureAccounting
class LRBinding(TrainerCallback):
 def __init__(self,resume):self.resume=resume
 def on_train_begin(self,args,state,control,optimizer=None,lr_scheduler=None,**kw):
  actual=optimizer
  while hasattr(actual,'optimizer'):actual=actual.optimizer
  scheduler=lr_scheduler
  while hasattr(scheduler,'scheduler'):scheduler=scheduler.scheduler
  assert isinstance(actual,torch.optim.AdamW), f'Unexpected optimizer: {type(actual).__module__}.{type(actual).__name__}';assert hasattr(scheduler,'optimizer');shared=all(x is y for x,y in zip(actual.param_groups,scheduler.optimizer.param_groups));assert len(actual.param_groups)==len(scheduler.get_last_lr())
  if self.resume:
   assert scheduler.last_epoch==state.global_step;scheduler.optimizer=actual
   for g,lr in zip(actual.param_groups,scheduler.get_last_lr()):g['lr']=lr
  self.actual,self.scheduler,self.output=actual,scheduler,Path(args.output_dir)
  if args.process_index==0:(self.output/'lr-binding.json').write_text(json.dumps({'resume':str(self.resume),'source_group_objects_shared':shared,'resume_binding_reconnected':bool(self.resume),'global_step':state.global_step,'optimizer_type':type(actual).__name__,'science_update_changed':False},indent=2)+'\n')
 def on_step_begin(self,args,state,control,**kw):
  x=[float(g['lr']) for g in self.actual.param_groups];y=[float(z) for z in self.scheduler.get_last_lr()];assert all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15) for a,b in zip(x,y))
  with (self.output/f'applied-lr-rank{args.process_index}.jsonl').open('a') as f:f.write(json.dumps({'update_step':state.global_step+1,'actual':x,'scheduler':y})+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--stop-at',type=int,required=True);p.add_argument('--resume',type=Path);a=p.parse_args();c=yaml.safe_load(a.config.read_text());assert c['full_determinism'] and c['num_train_epochs']==1 and c['learning_rate']==1e-5
 if a.resume:c['resume_from_checkpoint']=str(a.resume)
 from llamafactory.train.tuner import run_exp
 run_exp(c,callbacks=[ExposureAccounting(a.resume,a.stop_at),LRBinding(a.resume)])
if __name__=='__main__':main()
