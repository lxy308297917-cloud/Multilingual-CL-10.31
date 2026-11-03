"""Low-rank coordinates and paired functional cross-covariance, no new optimizer."""
import torch

def compact_update_modes(delta,basis,relative_tolerance=1e-7):
 """Exact small SVD of the fixed two-sided projected update (up to roundoff)."""
 x=delta.double();b=basis.to(device='cpu',dtype=torch.float64);l,r=b.left,b.right
 if b.rank==0:u,s,vh=torch.linalg.svd(x,full_matrices=False);v=vh.T
 elif b.left_free and b.right_free:
  z=l.T@x@r;us,s,vh=torch.linalg.svd(z,full_matrices=False);u=l@us;v=r@vh.T
 elif b.right_free:
  z=x@r;z=z-l@(l.T@z);u,s,vh=torch.linalg.svd(z,full_matrices=False);v=r@vh.T
 else:
  z=l.T@x;z=z-(z@r)@r.T;us,s,vh=torch.linalg.svd(z,full_matrices=False);u=l@us;v=vh.T
 keep=s>relative_tolerance*s[0] if s.numel() and float(s[0])>0 else torch.zeros_like(s,dtype=torch.bool)
 return u[:,keep].detach().float().contiguous(),s[keep].detach().float().contiguous(),v[:,keep].detach().float().contiguous()

def centered_cross_space(g1,g2,categories,max_rank=2,rank=None):
 """Per-module coefficient gradients, paired rows. No reliance on global means."""
 assert g1.shape==g2.shape and len(categories)==len(g1)
 if g1.shape[1]==0:return g1.new_empty(0,0),[]
 x=g1.double().clone();y=g2.double().clone()
 for category in sorted(set(categories)):
  ids=[i for i,c in enumerate(categories) if c==category];assert len(ids)>=2
  x[ids]-=x[ids].mean(0);y[ids]-=y[ids].mean(0)
 c=(x.T@y+y.T@x)/(2*max(1,len(x)-1));values,vectors=torch.linalg.eigh(c);order=torch.argsort(values,descending=True);values=values[order];vectors=vectors[:,order]
 if rank is None:
  threshold=1e-8*max(float(values.abs().max()),1e-30);rank=min(max_rank,int((values>threshold).sum()))
 assert 0<=rank<=g1.shape[1]
 return vectors[:,:rank].float().contiguous(),values.tolist()

def project_coefficient(direction,basis,reuse=False):
 shared=basis@(basis.T@direction)
 return shared if reuse else direction-shared
