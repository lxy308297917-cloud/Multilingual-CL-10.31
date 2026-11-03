"""Exact two-sided SST projectors, with thin-SVD rectangular complements.

Upstream: Arnav0400/SST ff0c5bfb32f621ca952d12042b046251181b8aa5.
No gradient transformation, approximate SVD, or dense identity projectors.
"""
from dataclasses import dataclass
import math
import torch

TARGET_MODULES=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')

def is_target(name):
 return any(name.endswith('.'+module+'.weight') for module in TARGET_MODULES)

@dataclass
class Subspace:
 shape:tuple
 rank:int
 left:torch.Tensor
 right:torch.Tensor
 left_free:bool
 right_free:bool

 @classmethod
 def from_weight(cls,weight,ratio=.99,driver=None):
  if weight.ndim!=2 or not 0<=ratio<=1:raise ValueError('Expected matrix and rank fraction in [0,1]')
  if weight.dtype not in (torch.float32,torch.float64):raise TypeError('SVD/reference must be FP32 or FP64')
  if not torch.isfinite(weight).all():raise ValueError('Nonfinite reference')
  m,n=weight.shape;k=math.floor(ratio*min(m,n))
  if k==0:return cls((m,n),0,weight.new_empty(m,0),weight.new_empty(n,0),False,False)
  u,_,vh=torch.linalg.svd(weight,full_matrices=False,**({'driver':driver} if driver else {}))
  # On the short axis the thin SVD already spans the entire ambient space.
  # Its tail is a complete free basis. On the long axis keep the protected
  # basis so the extra rectangular nullspace is never accidentally discarded.
  lf=m<=n;rf=n<=m
  left=u[:,k:] if lf else u[:,:k]
  right=vh[k:].T if rf else vh[:k].T
  return cls((m,n),k,left.contiguous(),right.contiguous(),lf,rf)

 def to(self,device=None,dtype=None):
  return Subspace(self.shape,self.rank,self.left.to(device=device,dtype=dtype),self.right.to(device=device,dtype=dtype),self.left_free,self.right_free)

 def project(self,delta):
  if tuple(delta.shape)!=tuple(self.shape):raise ValueError('Matrix shape mismatch')
  if self.rank==0:return delta.clone()
  l,r=self.left,self.right
  if l.device!=delta.device or l.dtype!=delta.dtype or r.device!=delta.device or r.dtype!=delta.dtype:raise TypeError('Basis and update precision/device must agree')
  if self.left_free and self.right_free:return l@((l.T@delta)@r)@r.T
  if self.right_free:
   y=delta@r;return (y-l@(l.T@y))@r.T
  y=l.T@delta;return l@(y-(y@r)@r.T)

 def leakage(self,delta):
  if self.rank==0:return {'left_absolute':0.,'right_absolute':0.,'left_relative':0.,'right_relative':0.}
  l,r=self.left,self.right
  a=delta-l@(l.T@delta) if self.left_free else l.T@delta
  b=delta-(delta@r)@r.T if self.right_free else delta@r
  norm=float(torch.linalg.vector_norm(delta));la=float(torch.linalg.vector_norm(a));lb=float(torch.linalg.vector_norm(b))
  return dict(left_absolute=la,right_absolute=lb,left_relative=la/max(norm,1e-30),right_relative=lb/max(norm,1e-30))

 def inventory(self):
  m,n=self.shape;k=self.rank
  return dict(shape=list(self.shape),protected_rank=k,left_free_dimension=m-k,right_free_dimension=n-k,free_space_dimension=(m-k)*(n-k),free_space_coordinate_ratio=(m-k)*(n-k)/(m*n),stored_left_shape=list(self.left.shape),stored_right_shape=list(self.right.shape),left_representation='free' if self.left_free else 'protected',right_representation='free' if self.right_free else 'protected')

class AnchoredState:
 """Cumulative fixed-basis projection; refresh after the projected true step."""
 def __init__(self,weights,ratio=.99,refresh_interval=None,driver=None):
  if refresh_interval is not None and refresh_interval<1:raise ValueError('Refresh interval must be positive')
  self.ratio=ratio;self.refresh_interval=refresh_interval;self.driver=driver;self.step=0;self.refresh_steps=[]
  self.references={k:v.detach().clone() for k,v in weights.items()}
  self.bases={k:Subspace.from_weight(v,ratio,driver) for k,v in self.references.items()}
  self.initial_references={k:v.clone() for k,v in self.references.items()}
  self.initial_bases={k:v.to() for k,v in self.bases.items()}

 @torch.no_grad()
 def project_proposal(self,proposed):
  if set(proposed)!=set(self.references):raise ValueError('Target matrix set changed')
  return {k:w.clone() if self.bases[k].rank==0 else self.references[k]+self.bases[k].project(w-self.references[k]) for k,w in proposed.items()}

 @torch.no_grad()
 def completed_step(self,projected):
  self.step+=1
  if self.refresh_interval and self.step%self.refresh_interval==0:
   self.references={k:v.detach().clone() for k,v in projected.items()}
   self.bases={k:Subspace.from_weight(v,self.ratio,self.driver) for k,v in self.references.items()}
   self.refresh_steps.append(self.step)

 def state_dict(self):
  return dict(version=1,ratio=self.ratio,refresh_interval=self.refresh_interval,step=self.step,refresh_steps=self.refresh_steps,references=self.references,bases=self.bases,initial_references=self.initial_references,initial_bases=self.initial_bases)

 @classmethod
 def from_state_dict(cls,state):
  if state.get('version')!=1:raise ValueError('Unsupported SST state version')
  x=object.__new__(cls);x.driver=None
  for key in ['ratio','refresh_interval','step','refresh_steps','references','bases','initial_references','initial_bases']:setattr(x,key,state[key])
  return x
