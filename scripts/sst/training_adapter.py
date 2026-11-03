"""Minimal LF/DeepSpeed adapter, wrapping the actual ZeRO optimizer step.

Accelerate calls DeepSpeedEngine.step inside backward; HF optimizer callbacks
are too late to capture pre-update masters. No framework source is modified.
"""
import hashlib,json,time,types
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import TrainerCallback
from projector import Subspace,is_target
from optimizer_adapter import DeepSpeedMaster

def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1024**2),b''):h.update(c)
 return h.hexdigest()
def dump(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')

class SSTCallback(TrainerCallback):
 def __init__(self,ratio=.99,refresh_interval=None,resume=None,svd_driver='gesvd',audit_moments=False,initial_cache=None,initial_cache_sha256=None):
  self.initial_cache=Path(initial_cache) if initial_cache else None;self.initial_cache_sha256=initial_cache_sha256;self.ratio=ratio;self.interval=refresh_interval;self.resume=Path(resume) if resume else None;self.driver=svd_driver;self.audit_moments=audit_moments;self.step=0;self.refresh_steps=[];self.references={};self.bases={}
 def log(self,name,row):
  if self.rank==0:
   with (self.output/name).open('a') as f:f.write(json.dumps(row)+'\n')
 def construct(self):
  t=time.monotonic();new_refs={};new_bases={}
  for i,(name,accessor) in enumerate(self.targets.items()):
   w=accessor.get();new_refs[name]=w.detach().cpu().clone();owner=i%self.world
   m,n=w.shape;k=int(self.ratio*min(m,n));lf=m<=n;rf=n<=m
   if self.rank==owner:b=Subspace.from_weight(w,self.ratio,self.driver if w.is_cuda else None)
   else:
    b=Subspace((m,n),k,torch.empty(m,(min(m,n)-k) if lf else k,device=w.device),torch.empty(n,(min(m,n)-k) if rf else k,device=w.device),lf,rf)
   if self.world>1:
    dist.broadcast(b.left,src=owner);dist.broadcast(b.right,src=owner)
   new_bases[name]=b.to(device='cpu');del w,b
   if self.rank==0 and (i+1)%28==0:print('[SST] SVD matrices %s/%s at step %s'%(i+1,len(self.targets),self.step),flush=True)
  self.references,self.bases=new_refs,new_bases
  torch.cuda.synchronize();self.log('svd-cost.jsonl',{'step':self.step,'seconds':time.monotonic()-t,'matrices':len(new_bases),'svd_driver':self.driver})
 def on_train_begin(self,args,state,control,model=None,optimizer=None,lr_scheduler=None,**kw):
  self.output=Path(args.output_dir);self.output.mkdir(parents=True,exist_ok=True);self.rank=dist.get_rank() if dist.is_initialized() else 0;self.world=dist.get_world_size() if dist.is_initialized() else 1
  zero=optimizer
  while hasattr(zero,'optimizer') and not hasattr(zero,'get_full_hp_param') and not hasattr(zero,'bit16_groups'):zero=zero.optimizer
  if not hasattr(zero,'optimizer'):raise RuntimeError('Cannot identify actual DeepSpeed ZeRO optimizer')
  self.zero=zero;self.stage=3 if hasattr(zero,'fp32_partitioned_groups_flat') else 2
  if getattr(zero,'swap_optimizer',False):raise RuntimeError('NVMe swap unsupported')
  physical=model
  while hasattr(physical,'module'):physical=physical.module
  selected={n:p for n,p in physical.named_parameters() if is_target(n)}
  expected=7*physical.config.num_hidden_layers
  if len(selected)!=expected:raise RuntimeError('Missing target matrices: %s != %s'%(len(selected),expected))
  self.targets={n:DeepSpeedMaster(p) for n,p in selected.items()}
  self.all_accessors={n:DeepSpeedMaster(p) for n,p in physical.named_parameters() if p.requires_grad}
  self.unconstrained=[{'name':n,'elements':getattr(p,'ds_numel',p.numel()),'trainable':p.requires_grad} for n,p in physical.named_parameters() if n not in selected]
  if self.resume:
   saved=torch.load(self.resume/'sst-state.pt',map_location='cpu',weights_only=False)
   if saved['ratio']!=self.ratio or saved['interval']!=self.interval or saved['step']!=state.global_step:raise RuntimeError('SST checkpoint protocol/step mismatch')
   self.step=saved['step'];self.refresh_steps=saved['refresh_steps'];self.references=saved.get('references');self.bases=saved.get('bases');self.initial_path=Path(saved['initial_artifact']);assert file_sha(self.initial_path)==saved['initial_sha256'];initial=torch.load(self.initial_path,map_location='cpu',weights_only=False)
   self.initial_references=initial['references'];self.initial_bases=initial['bases']
   if self.references is None:self.references=self.initial_references.copy();self.bases=self.initial_bases.copy()
   restore_accessors=self.all_accessors if saved.get('version',1)>=2 else self.targets
   if set(saved['master_sha256'])!=set(restore_accessors):raise RuntimeError('Master checkpoint coverage mismatch')
   for name,a in restore_accessors.items():
    if tensor_sha(a.get())!=saved['master_sha256'][name]:raise RuntimeError('Restored FP32 master mismatch: '+name)
   self.log('resume-integrity.jsonl',{'step':self.step,'all_master_weights_exact':True,'matrices':len(self.targets),'all_unique_parameters_checked':len(restore_accessors),'state_sha256':file_sha(self.resume/'sst-state.pt')})
  else:
   if self.initial_cache:
    if file_sha(self.initial_cache)!=self.initial_cache_sha256:raise RuntimeError('Initial basis cache file hash mismatch')
    initial=torch.load(self.initial_cache,map_location='cpu',weights_only=False)
    if set(initial['references'])!=set(self.targets) or set(initial['bases'])!=set(self.targets):raise RuntimeError('Initial basis cache target mismatch')
    for n,a in self.targets.items():
     current=a.get();b=initial['bases'][n]
     if tensor_sha(current)!=tensor_sha(initial['references'][n]):raise RuntimeError('Initial master/reference cache mismatch: '+n)
     if b.rank!=int(self.ratio*min(current.shape)) or tuple(b.shape)!=tuple(current.shape):raise RuntimeError('Initial cache rank/shape mismatch: '+n)
    self.references,self.bases=initial['references'],initial['bases'];self.initial_path=self.initial_cache
    self.log('svd-cost.jsonl',{'step':0,'cache_hit':True,'exact_master_hashes_verified':len(self.targets),'cache_sha256':self.initial_cache_sha256})
   else:
    self.construct();self.initial_path=self.output/'sst-initial.pt'
    if self.rank==0:torch.save({'references':self.references,'bases':self.bases},self.initial_path)
   self.initial_references=self.references.copy();self.initial_bases=self.bases.copy()
   if self.world>1:dist.barrier()
  self.initial_sha256=file_sha(self.initial_path)
  scheduler=lr_scheduler
  while hasattr(scheduler,'scheduler'):scheduler=scheduler.scheduler
  base=zero.optimizer
  if hasattr(scheduler,'optimizer'):
   if len(scheduler.get_last_lr())!=len(base.param_groups):raise RuntimeError('Scheduler/master group mismatch')
   scheduler.optimizer=base
   if self.resume:
    for g,lr in zip(base.param_groups,scheduler.get_last_lr()):g['lr']=lr
  if self.rank==0:
   dump(self.output/'subspace-inventory.json',{'matrices':{n:b.inventory() for n,b in self.bases.items()},'unconstrained_parameters':self.unconstrained,'stage':self.stage,'actual_optimizer':type(base).__name__,'trainable_unique_parameters':sum(getattr(p,'ds_numel',p.numel()) for p in physical.parameters() if p.requires_grad),'initial_sha256':self.initial_sha256})
  original=zero.step
  def actual_step(*aa,**kk):return self.projected_step(original,*aa,**kk)
  zero.step=actual_step;self.original_step=original;self.started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 @torch.no_grad()
 def projected_step(self,original,*aa,**kk):
  started=time.monotonic();before={n:a.get().detach().cpu().clone() for n,a in self.targets.items()}
  ret=original(*aa,**kk)
  if bool(getattr(self.zero,'overflow',False)):raise RuntimeError('Nonfinite/overflow skipped true optimizer update')
  self.step+=1
  for name,a in self.targets.items():
   current=a.get();old=before.pop(name).to(current.device);reference=self.references[name].to(current.device);basis=self.bases[name].to(device=current.device);raw=current-old
   # Full original gradient and Adam moments have already been consumed.
   if self.audit_moments:
    from deepspeed.utils import safe_get_full_optimizer_state
    moments={k:safe_get_full_optimizer_state(a.p,k).clone() for k in ['exp_avg','exp_avg_sq']}
   projected=current.clone() if self.ratio==0 else reference+basis.project(current-reference)
   if not torch.isfinite(projected).all():raise RuntimeError('Nonfinite projected master: '+name)
   a.set(projected)
   if self.audit_moments:
    for k,v in moments.items():
     if not torch.equal(v,safe_get_full_optimizer_state(a.p,k)):raise RuntimeError('Projection changed Adam moment '+name)
   if self.audit_moments or self.step in [1,100,200,300,400]:
    if not torch.equal(a.get(),projected):raise RuntimeError('FP32 master writeback failed: '+name)
   actual=projected-old;cumulative=projected-reference;initial_basis=self.initial_bases[name].to(device=current.device);initial_delta=projected-self.initial_references[name].to(current.device)
   raw_norm=float(raw.norm());applied_norm=float(actual.norm());leak=basis.leakage(cumulative);initial_leak=initial_basis.leakage(initial_delta)
   self.log('matrix-updates.jsonl',{'step':self.step,'name':name,'raw_adam_update_norm':raw_norm,'applied_update_norm':applied_norm,'removed_update_norm_ratio':float((raw-actual).norm())/max(raw_norm,1e-30),'numerically_changed_master_elements':int((projected!=old).sum()),'current_reference_step':self.refresh_steps[-1] if self.refresh_steps else 0,'current_reference_leakage':leak,'initial_reference_leakage':initial_leak,'master_verified':self.audit_moments or self.step in [1,100,200,300,400]})
   del current,old,reference,basis,raw,projected,actual,cumulative,initial_basis,initial_delta
  projection_finished=time.monotonic();refresh=False
  if self.interval and self.step%self.interval==0:
   self.construct();self.refresh_steps.append(self.step);refresh=True
  torch.cuda.synchronize();self.log('step-cost.jsonl',{'step':self.step,'optimizer_and_projection_seconds':projection_finished-started,'total_seconds_including_refresh':time.monotonic()-started,'refreshed':refresh,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
  return ret
 def save_constraint(self,path):
  hashes={n:tensor_sha(a.get()) for n,a in self.all_accessors.items()}
  if self.rank==0:
   torch.save({'version':2,'ratio':self.ratio,'interval':self.interval,'step':self.step,'refresh_steps':self.refresh_steps,'references':self.references if self.refresh_steps else None,'bases':self.bases if self.refresh_steps else None,'initial_artifact':str(self.initial_path),'initial_sha256':self.initial_sha256,'master_sha256':hashes},Path(path)/'sst-state.pt')
  if self.world>1:dist.barrier()
 def on_save(self,args,state,control,**kw):
  if self.step!=state.global_step:raise RuntimeError('True optimizer/HF step mismatch')
  self.save_constraint(self.output/('checkpoint-'+str(state.global_step)))
 def on_train_end(self,args,state,control,**kw):
  if self.step!=state.global_step:raise RuntimeError('Final true optimizer/HF step mismatch')
  self.save_constraint(self.output)
  self.log('training-resource-summary.jsonl',{'step':self.step,'elapsed_seconds':time.monotonic()-self.started,'refresh_steps':self.refresh_steps,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
  self.zero.step=self.original_step
