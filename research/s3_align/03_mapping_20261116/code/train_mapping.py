"""LLaMA-Factory main CE; frozen English uniform36 keep; fixed mapping alignment."""
from common import *
import argparse,sys,os,time,contextlib,math
import torch,yaml,numpy as np
from safetensors.torch import load_file
torch.set_num_threads(4)
@contextlib.contextmanager
def eval_grad(model):
 flags=[(m,m.training) for m in model.modules()];model.eval()
 try:yield
 finally:
  for m,flag in flags:m.training=flag
def body(model,row):
 raw=model.get_base_model() if hasattr(model,'get_base_model') else model
 ids=torch.tensor([row['input_ids']],device=next(model.parameters()).device);saved=[]
 def hook(mod,args,out):saved.append((out[0] if isinstance(out,tuple) else out)[0,row['body_positions']].float().mean(0))
 handle=raw.model.layers[17].register_forward_hook(hook)
 try:raw.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
 finally:handle.remove()
 assert len(saved)==1 and saved[0].requires_grad
 return saved[0]
def alignment(model,row,index,arm,maps):
 h=body(model,row);device=h.device
 z=((h-maps['body.mean.sw'].to(device))@maps[f'{arm}.projection.sw'].to(device))/maps[f'{arm}.std.sw'].to(device)
 target=maps[f'{arm}.teacher'][index].to(device)
 assert not target.requires_grad
 return (z-target).square().mean()
def gradnorm(loss,params):
 grads=torch.autograd.grad(loss,params,allow_unused=True)
 assert all(torch.isfinite(g).all() for g in grads if g is not None)
 return math.sqrt(sum(float(g.double().square().sum()) for g in grads if g is not None))
class CalibrationComplete(Exception):pass
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['T0','T1','T2'],required=True);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--calibrate',action='store_true');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 cfg=yaml.safe_load(a.config.read_text());out=Path(cfg['output_dir']);assert not out.exists(),out
 assert cfg['per_device_train_batch_size']*cfg['gradient_accumulation_steps']==16
 assert cfg['use_reentrant_gc'] is False and cfg['seed']==42 and not cfg.get('adapter_name_or_path')
 if a.calibrate:
  cfg['output_dir']=str(CK/'calibration-v1');out=Path(cfg['output_dir']);assert not out.exists()
 metadata=read(R/'assets/mapping_metadata.json')
 for p,d in metadata['hashes'].items():assert sha(p)==d,p
 if not a.calibrate and not a.smoke:
  lock=read(R/'TRAINING_LOCK.json')
  for p,d in lock['files'].items():assert sha(p)==d,p
 pool=rows(R/'inputs/alignment.jsonl');maps=load_file(str(R/'assets/mappings.safetensors'))
 sys.path.insert(0,str(W/'work/s3-align-v0-20261113'))
 from keep_loss import load_keep_assets,uniform_keep_loss
 keep=load_keep_assets();assert keep.normalizers==metadata['keep_normalizers']
 sys.path.insert(0,str(P/'scripts/sft_v1'))
 from train_formal import ExposureAccounting
 import llamafactory.train.sft.workflow as workflow
 Original=workflow.CustomSeq2SeqTrainer;holder={};start=time.monotonic()
 calibration=None if a.calibrate else read(R/'assets/gradient_calibration.json')
 class Joint(Original):
  def __init__(self,*args,**kw):
   super().__init__(*args,**kw);holder['trainer']=self;self.last_aux=-1;self.aux=[]
  def training_step(self,model,inputs,num_items_in_batch=None):
   result=super().training_step(model,inputs,num_items_in_batch=num_items_in_batch)
   assert self.accelerator.gradient_accumulation_steps==1
   step=int(self.state.global_step)
   if step==self.last_aux:return result
   self.last_aux=step
   with self.compute_loss_context_manager():loss,info=uniform_keep_loss(model,keep,step)
   self.accelerator.backward(loss)
   rec=dict(step=step+1,keep_id=info['id'],keep_input_tokens=info['input_tokens'],keep_loss=float(loss.detach()),keep_scale=info['normalizer'],mu=1.)
   total=result+loss.detach();del loss
   if a.arm!='T0':
    i=step%1024;row=pool[i]['encoded']['sw']
    with eval_grad(model),self.compute_loss_context_manager():
     raw=alignment(model,row,i,a.arm,maps);loss=calibration['lambda'][a.arm]*raw
     assert loss.requires_grad and torch.isfinite(loss)
     # Sparse fixed update indices record actual alignment gradient norms, without dynamic rescaling.
     if step in [0,100,500,2000,4000]:
      params=[p for p in model.parameters() if p.requires_grad]
      grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
      rec['alignment_gradient_norm']=math.sqrt(sum(float(g.double().square().sum()) for g in grads if g is not None));del grads
     self.accelerator.backward(loss)
    rec.update(alignment_id=pool[i]['id'],alignment_input_tokens=len(row['input_ids']),alignment_raw=float(raw.detach()),alignment_weighted=float(loss.detach()),lambda_fixed=calibration['lambda'][a.arm])
    total+=loss.detach();del loss,raw
   self.aux.append(rec)
   with (out/'auxiliary_live.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   return total
 class Counter(ExposureAccounting):
  def on_train_begin(self,args,state,control,model=None,**kw):
   if a.calibrate:
    out.mkdir(parents=True,exist_ok=True);model.save_pretrained(out/'initial_adapter',safe_serialization=True)
    ds=holder['trainer'].train_dataset;source=rows(DATA/'sw_train.jsonl');assert len(ds)==len(source)==65415
    lookup={x['id']:i for i,x in enumerate(source)};measurements=[];params=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==33030144
    initial=[p.detach().cpu().clone() for p in params]
    for i in metadata['calibration_indices']:
     r=pool[i];sample=ds[lookup[r['id']]]
     with eval_grad(model),torch.autocast('cuda',dtype=torch.bfloat16):
      batch={k:torch.tensor([sample[k]],device=next(model.parameters()).device) for k in ['input_ids','attention_mask','labels']}
      ce=model(**batch,use_cache=False).loss;ceval=float(ce.detach());nce=gradnorm(ce,params);del ce,batch
      record=dict(id=r['id'],alignment_index=i,ce_loss=ceval,ce_norm=nce,input_tokens=len(sample['input_ids']))
      for arm in ['T1','T2']:
       loss=alignment(model,r['encoded']['sw'],i,arm,maps);value=float(loss.detach());norm=gradnorm(loss,params);del loss
       assert math.isfinite(norm) and norm>1e-8
       record[arm+'_raw_loss']=value;record[arm+'_norm']=norm
     measurements.append(record);print(json.dumps(record),flush=True)
    assert all(torch.equal(p.detach().cpu(),old) for p,old in zip(params,initial))
    med={k:float(np.median([r[k+'_norm'] for r in measurements])) for k in ['ce','T1','T2']}
    coefficients={k:.1*med['ce']/med[k] for k in ['T1','T2']}
    assert all(0<v<=10 and math.isfinite(v) for v in coefficients.values()),coefficients
    write(R/'assets/gradient_calibration.json',dict(at=now(),status='NO_UPDATE_CALIBRATION_COMPLETE',rows=measurements,median_norms=med,lambda_=coefficients,**{'lambda':coefficients},target_ratio=.1,mu=1.,all_parameters_bitwise_unchanged=True,seed=42,dropout=False,init_adapter_sha256=sha(out/'initial_adapter/adapter_model.safetensors'),seconds=time.monotonic()-start,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']))
    raise CalibrationComplete()
   super().on_train_begin(args,state,control,model=model,**kw)
   assert args.world_size==1 and state.max_steps==(8 if a.smoke else 4089)
   assert self.parameters['trainable_unique_parameters']==33030144
   model.save_pretrained(out/'initial_adapter',safe_serialization=True)
   assert sha(out/'initial_adapter/adapter_model.safetensors')==calibration['init_adapter_sha256']
   torch.cuda.reset_peak_memory_stats();self.start=time.monotonic()
  def record(self,args,state,path):
   super().record(args,state,path);jsonl(path/'auxiliary_exposure.jsonl',holder['trainer'].aux)
 workflow.CustomSeq2SeqTrainer=Joint
 from llamafactory.train.tuner import run_exp
 counter=Counter()
 try:run_exp(cfg,callbacks=[counter])
 except CalibrationComplete:return
 stats=read(out/'accounting-rank0.json');trace=rows(out/'exposure-rank0.jsonl');aux=holder['trainer'].aux
 assert stats['trace_covers_all_local_samples'] and len(trace)==(128 if a.smoke else 65415)
 assert len(aux)==(8 if a.smoke else 4089)
 if not a.smoke:
  assert stats['global_counts']['input_tokens']==19135489
  assert stats['global_counts']['shifted_assistant_tokens']==14314363
 tensors=load_file(str(out/'adapter_model.safetensors'));assert all(torch.isfinite(x).all() for x in tensors.values())
 write(out/'TRAINING_DONE.json',dict(at=now(),status='SMOKE_COMPLETE' if a.smoke else 'TRAINING_DONE',arm=a.arm,budget='SMOKE' if a.smoke else 'FULL-65K',steps=stats['step'],counts=stats['global_counts'],weights_sha256={'adapter_model.safetensors':sha(out/'adapter_model.safetensors')},initial_sha256=sha(out/'initial_adapter/adapter_model.safetensors'),exposure_sha256=sha(out/'exposure-rank0.jsonl'),auxiliary_exposure_sha256=sha(out/'auxiliary_exposure.jsonl'),seconds=time.monotonic()-start,training_seconds=time.monotonic()-counter.start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],config_sha256=sha(a.config)))
 print(json.dumps(read(out/'TRAINING_DONE.json')),flush=True)
if __name__=='__main__':main()
