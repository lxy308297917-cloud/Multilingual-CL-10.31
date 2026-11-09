"""Explicit covariance definitions with small Gram matrices and float64 arithmetic."""
import torch

def covariance_difference(base,adapted):
    """Rows are observations; unbiased centered covariance, same row identities."""
    assert base.shape==adapted.shape and base.ndim==2 and len(base)>1
    x=base.double();y=adapted.double();x=x-x.mean(0);y=y-y.mean(0);den=len(x)-1
    gx=x@x.T;gy=y@y.T;cross=y@x.T
    numerator=gy.square().sum()+gx.square().sum()-2*cross.square().sum()
    roundoff=1e-12*(gy.square().sum()+gx.square().sum()).clamp_min(1)
    assert numerator>=-roundoff
    fro=float(numerator.clamp_min(0).sqrt()/den);base_fro=float(gx.square().sum().sqrt()/den)
    z=torch.cat([y,x]);gram=z@z.T;v,u=torch.linalg.eigh(gram);root=(u*v.clamp_min(0).sqrt())@u.T
    sign=torch.cat([torch.ones(len(x)), -torch.ones(len(x))]).double()
    signed=(root*sign.unsqueeze(0))@root
    spectral=float(torch.linalg.eigvalsh((signed+signed.T)/2).abs().max()/den)
    base_spectral=float(torch.linalg.eigvalsh(gx).clamp_min(0).max()/den)
    if torch.equal(base,adapted):fro=0.;spectral=0.
    return dict(frobenius=fro,base_frobenius=base_fro,normalized_frobenius=fro/max(base_fro,1e-12),spectral=spectral,base_spectral=base_spectral,normalized_spectral=spectral/max(base_spectral,1e-12),observations=len(x))

def normalized_drift(base,adapted):
    assert base.shape==adapted.shape and base.ndim==3
    # positions x layers x hidden; one per-layer observation for this example.
    return (adapted.double()-base.double()).square().sum((0,2))/base.double().square().sum((0,2)).clamp_min(1e-12)

def cosine(x,y):
    x=x.double();y=y.double()
    return (x*y).sum(-1)/(x.norm(dim=-1)*y.norm(dim=-1)).clamp_min(1e-12)

def validate():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(713)
    for n,d in [(5,13),(13,5),(4,1)]:
        x=torch.randn(n,d,generator=g,dtype=torch.float64);y=x+.2*torch.randn(n,d,generator=g,dtype=torch.float64)
        cx=x-x.mean(0);cy=y-y.mean(0);diff=(cy.T@cy-cx.T@cx)/(n-1)
        r=covariance_difference(x,y)
        assert abs(r['frobenius']-float(diff.norm()))<1e-10
        assert abs(r['spectral']-float(torch.linalg.matrix_norm(diff,ord=2)))<1e-10
        shift=torch.randn(d,generator=g,dtype=torch.float64)
        translated=covariance_difference(x+shift,y+shift)
        assert abs(r['normalized_frobenius']-translated['normalized_frobenius'])<1e-10
        assert covariance_difference(x,x)['frobenius']==covariance_difference(x,x)['spectral']==0
        scaled=covariance_difference(x,2*x)
        assert abs(scaled['normalized_frobenius']-3)<1e-10
        assert abs(scaled['normalized_spectral']-3)<1e-10
    a=torch.randn(4,36,11,generator=g);assert torch.equal(normalized_drift(a,a),torch.zeros(36,dtype=torch.float64))
    assert torch.allclose(cosine(a,a),torch.ones(4,36,dtype=torch.float64),atol=1e-12)
    print('CPU oracles passed: Gram Frobenius/spectral vs dense, translation/scale, zero drift and cosine identity; no GPU')
if __name__=='__main__':validate()
