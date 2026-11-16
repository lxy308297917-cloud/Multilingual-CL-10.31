"""Frozen PCA128 / ridge0.1 / CCA64 mapping equations, extracted unchanged."""
import torch

def pca(x):return torch.linalg.svd(x,full_matrices=False)[2][:128].T

def cca(x,y):
 qx,qy=pca(x),pca(y);zx,zy=x@qx,y@qy
 def white(z):
  c=z.T@z/(len(z)-1);e,v=torch.linalg.eigh(c+.1*c.trace()/128*torch.eye(128,dtype=torch.float64))
  return (v*e.rsqrt())@v.T
 wx,wy=white(zx),white(zy);u,s,v=torch.linalg.svd(wx@(zx.T@zy/(len(x)-1))@wy)
 return qx@wx@u[:,:64],qy@wy@v.T[:,:64],s
