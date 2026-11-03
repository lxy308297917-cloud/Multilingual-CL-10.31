"""Project actual optimizer displacements; never silently fall back from masters."""
import torch
from projector import AnchoredState

class DeepSpeedMaster:
 def __init__(self,parameter):self.p=parameter
 def get(self):
  from deepspeed.utils import safe_get_full_fp32_param
  w=safe_get_full_fp32_param(self.p)
  if w is None or w.dtype!=torch.float32:raise RuntimeError('SST requires accessible full FP32 master weight')
  return w
 @torch.no_grad()
 def set(self,value):
  from deepspeed.utils import safe_set_full_fp32_param
  p=self.p
  if value.dtype!=torch.float32:raise TypeError('Master write must be FP32')
  safe_set_full_fp32_param(p,value)
  if hasattr(p,'ds_id'):
   # safe_set updates ONLY FP32 optimizer state. Also synchronize the ZeRO3
   # low-precision partition from which subsequent parameter gathers read.
   import torch.distributed as dist
   opt=p._z3_optimizer;group=opt.dp_process_group;rank=dist.get_rank(group)
   part=p.ds_tensor;start=rank*part.numel();available=max(0,min(part.numel(),value.numel()-start))
   part.zero_()
   if available:part[:available].copy_(value.flatten()[start:start+available])
   if p.numel():
    if p.numel()!=value.numel():raise RuntimeError('Unexpected available ZeRO3 parameter shape')
    p.data.copy_(value.to(p.dtype))
   if getattr(opt,'swap_optimizer',False):raise RuntimeError('NVMe optimizer swapping not validated')
   opt.invalidate_secondary_tensor()
  else:
   if not hasattr(p,'_hp_mapping'):raise RuntimeError('Missing ZeRO master mapping')
   p.data.copy_(value.to(device=p.device,dtype=p.dtype))

class TorchSST:
 """Small-model reference adapter used for actual Adam and resume tests."""
 def __init__(self,optimizer,parameters,ratio=.99,refresh_interval=None,state=None):
  self.optimizer=optimizer;self.parameters=parameters
  self.constraint=state or AnchoredState({k:p.detach() for k,p in parameters.items()},ratio,refresh_interval)
 @torch.no_grad()
 def step(self):
  before={k:p.detach().clone() for k,p in self.parameters.items()}
  result=self.optimizer.step()
  raw={k:p.detach().clone()-before[k] for k,p in self.parameters.items()}
  projected=self.constraint.project_proposal({k:p.detach() for k,p in self.parameters.items()})
  for k,p in self.parameters.items():p.copy_(projected[k])
  self.constraint.completed_step(projected)
  return result,raw
