"""Same-input activation capture/patching. No cross-language token alignment."""
from contextlib import contextmanager
import torch


def output_tensor(output):
    value=output[0] if isinstance(output,tuple) else output
    if not isinstance(value,torch.Tensor) or value.ndim!=3:raise ValueError('Expected [batch, tokens, hidden] module output')
    return value


@contextmanager
def capture(model,names,positions=None):
    modules=dict(model.named_modules());names=list(names)
    if not names or len(set(names))!=len(names) or not set(names).issubset(modules):raise ValueError('Require unique existing module names')
    if model.training:raise ValueError('Require eval mode')
    data={};handles=[]
    def hook_for(name):
        def hook(module,args,output):
            if name in data:raise ValueError('Repeated module call; disable generation cache and use one fixed-input forward')
            value=output_tensor(output)
            if positions is not None:
                if not positions or min(positions)<0 or max(positions)>=value.shape[1]:raise ValueError('Position out of sequence')
                value=value[:,positions,:]
            data[name]=value.detach().float().cpu().clone()
        return hook
    try:
        for name in names:handles.append(modules[name].register_forward_hook(hook_for(name)))
        yield data
        if set(data)!=set(names):raise ValueError('Some requested modules were not called')
    finally:
        for handle in handles:handle.remove()


@contextmanager
def patch(model,name,donor,positions=None):
    """One module output on identical target/donor input token positions only."""
    modules=dict(model.named_modules())
    if name not in modules or model.training:raise ValueError('Missing module or train mode')
    calls=[]
    def hook(module,args,output):
        if calls:raise ValueError('Only a single fixed-input forward is supported')
        calls.append(1);value=output_tensor(output)
        replacement=donor.to(device=value.device,dtype=value.dtype)
        if positions is None:
            if replacement.shape!=value.shape:raise ValueError('Donor shape mismatch; token alignment is required')
            new=replacement.clone()
        else:
            if not positions or len(set(positions))!=len(positions) or min(positions)<0 or max(positions)>=value.shape[1]:raise ValueError('Invalid patch positions')
            if replacement.shape!=(value.shape[0],len(positions),value.shape[-1]):raise ValueError('Aligned donor shape mismatch')
            new=value.clone();new[:,positions,:]=replacement
        if isinstance(output,tuple):return (new,*output[1:])
        return new
    handle=modules[name].register_forward_hook(hook)
    try:
        yield
        if not calls:raise ValueError('Patched module was not called')
    finally:handle.remove()


def compare_same_input(base,final):
    if base.shape!=final.shape:raise ValueError('Same-input shapes must agree')
    x=base.double().reshape(-1,base.shape[-1]);y=final.double().reshape(-1,final.shape[-1])
    cos=torch.nn.functional.cosine_similarity(x,y,dim=-1)
    return {'mean_token_cosine':float(cos.mean()),'relative_l2':float(torch.linalg.vector_norm(y-x)/(torch.linalg.vector_norm(x)+1e-12)),'positions':x.shape[0]}


def linear_cka(x,y):
    # Rows are aligned independent sample summaries, NOT mismatched language tokens.
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<2:raise ValueError('CKA requires at least two aligned samples')
    x=x.double()-x.double().mean(0);y=y.double()-y.double().mean(0)
    a=x@x.T;b=y@y.T;den=torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
    return float((a*b).sum()/den) if den>0 else None
