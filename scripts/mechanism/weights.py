"""Explicit effective-weight loading and reversible diagnostic interventions.

No training, downloads, quantization, or checkpoint writes. LoRA uses PEFT's
actual merge rule in an explicit common analysis dtype, not A/B subtraction.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import torch
from torch import nn


def load_pair(base_path, final_path, kind, dtype='float32', device='cpu'):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    if kind not in ('full','lora'):raise ValueError('kind must be full or lora')
    for path in (base_path,final_path):
        if not Path(path).is_dir():raise FileNotFoundError(path)
    if dtype not in ('float32','bfloat16'):raise ValueError('Unsupported analysis dtype')
    dt=getattr(torch,dtype)
    kwargs=dict(torch_dtype=dt,local_files_only=True,attn_implementation='eager')
    base=AutoModelForCausalLM.from_pretrained(base_path,**kwargs).eval()
    if kind=='full':final=AutoModelForCausalLM.from_pretrained(final_path,**kwargs).eval()
    else:
        backbone=AutoModelForCausalLM.from_pretrained(base_path,**kwargs).eval()
        adapter=PeftModel.from_pretrained(backbone,final_path,is_trainable=False,local_files_only=True)
        if any(getattr(m,'is_loaded_in_4bit',False) or getattr(m,'is_loaded_in_8bit',False) for m in adapter.modules()):raise ValueError('Quantized adapter unsupported')
        final=adapter.merge_and_unload(safe_merge=True).eval()
    base.requires_grad_(False);final.requires_grad_(False)
    final.to(device)
    return base,final


def parameters(model):
    if getattr(model,'is_quantized',False) or any(hasattr(m,'lora_A') for m in model.modules()):raise ValueError('Require unquantized effective-weight model; merge PEFT first')
    return dict(model.named_parameters(remove_duplicate=False))


def digest_tensor(tensor):
    t=tensor.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def unique_entries(model):
    aliases={};params=parameters(model)
    for name,p in params.items():aliases.setdefault(id(p),[]).append(name)
    return [(names[0],params[names[0]],names) for names in aliases.values()]


def validate_pair(base,final):
    b=parameters(base);f=parameters(final)
    if b.keys()!=f.keys():raise ValueError('Base/Final parameter names differ')
    for name in b:
        if b[name].shape!=f[name].shape or b[name].dtype!=f[name].dtype:raise ValueError(f'Shape/dtype mismatch: {name}')
        if not b[name].is_floating_point() or b[name].layout!=torch.strided:raise ValueError(f'Unsupported parameter {name}')
    bg={tuple(names) for _,_,names in unique_entries(base)};fg={tuple(names) for _,_,names in unique_entries(final)}
    if bg!=fg:raise ValueError('Tied parameter aliases differ')
    return b,f


def discover_components(model):
    """Detect actual decoder ModuleList, Attention/MLP and available projections."""
    candidates=[]
    for prefix,mod in model.named_modules():
        if not isinstance(mod,nn.ModuleList) or not len(mod):continue
        blocks=[]
        for i,block in enumerate(mod):
            children=dict(block.named_children())
            attention=[n for n,m in children.items() if 'attention' in type(m).__name__.lower()]
            mlp=[n for n,m in children.items() if 'mlp' in type(m).__name__.lower()]
            if len(attention)!=1 or len(mlp)!=1:break
            bp=f'{prefix}.{i}';ap=bp+'.'+attention[0];mp=bp+'.'+mlp[0]
            projections={n:ap+'.'+n for n,m in children[attention[0]].named_children() if isinstance(m,nn.Linear)}
            projections.update({n:mp+'.'+n for n,m in children[mlp[0]].named_children() if isinstance(m,nn.Linear)})
            blocks.append({'index':i,'block':bp,'attention':ap,'mlp':mp,'projections':projections})
        if len(blocks)==len(mod):candidates.append(blocks)
    if len(candidates)!=1:raise ValueError('Cannot uniquely detect decoder layer structure')
    return candidates[0]


def select_parameters(model,layers,component):
    structure=discover_components(model);prefixes=[]
    if not layers or len(set(layers))!=len(layers):raise ValueError('Require unique layer indices')
    for i in layers:
        if i<0 or i>=len(structure):raise ValueError('Layer index outside model')
        item=structure[i]
        if component in ('block','attention','mlp'):prefixes.append(item[component])
        elif component in item['projections']:prefixes.append(item['projections'][component])
        else:raise ValueError(f'Undetected component {component} at layer {i}')
    selected={n for n in parameters(model) if any(n.startswith(p+'.') for p in prefixes)}
    if not selected:raise ValueError('Empty selection')
    for _,_,aliases in unique_entries(model):
        overlap=selected.intersection(aliases)
        if overlap and overlap!=set(aliases):raise ValueError('Selection splits tied weights')
    return selected


def drift_rows(base,final,chunk_elements=1048576):
    b,f=validate_pair(base,final);rows=[]
    for name,_,aliases in unique_entries(final):
        x=b[name].detach().cpu().reshape(-1);y=f[name].detach().cpu().reshape(-1)
        norm=diff=0.;changed=0
        for start in range(0,x.numel(),chunk_elements):
            a=x[start:start+chunk_elements].double();z=y[start:start+chunk_elements].double();d=z-a
            if not torch.isfinite(a).all() or not torch.isfinite(z).all():raise ValueError(f'Nonfinite weight {name}')
            norm+=float(a.square().sum());diff+=float(d.square().sum());changed+=int((z!=a).sum())
        rows.append({'parameter':name,'aliases':aliases,'elements':x.numel(),'changed_elements':changed,'changed_fraction':changed/x.numel(),'base_norm':norm**.5,'delta_norm':diff**.5,'relative_norm':diff**.5/(norm**.5+1e-12)})
    return rows


@contextmanager
def intervene(base,final,selected,mode='restore',alpha=0.,delta_norm=None):
    """Temporarily edit Final in memory, then restore byte-exactly even on error.

    restore: Base + alpha*delta in G, Final elsewhere.
    inject: Base + alpha*delta in G, Base elsewhere (use alpha=1).
    delta_norm rescales removal in restore mode, or injection in inject mode.
    At restore alpha=0 or inject alpha=1 the perturbation norm matches the
    requested value (up to dtype rounding). This is NOT a literal module swap.
    """
    b,f=validate_pair(base,final)
    if mode not in ('restore','inject') or not 0<=alpha<=1:raise ValueError('Invalid intervention')
    selected=set(selected)
    if not selected or not selected.issubset(f):raise ValueError('Unknown or empty parameter selection')
    entries=unique_entries(final)
    for _,_,aliases in entries:
        if selected.intersection(aliases) and not set(aliases).issubset(selected):raise ValueError('Partial tied selection')
    touched=[(n,p) for n,p,aliases in entries if mode=='inject' or n in selected]
    backups={n:p.detach().cpu().clone() for n,p in touched}
    hashes={n:digest_tensor(t) for n,t in backups.items()}
    original_norm=sum(float((backups[n].double()-b[n].detach().cpu().double()).square().sum()) for n in backups if n in selected)**.5
    scale=1.
    if delta_norm is not None:
        if delta_norm<0 or not torch.isfinite(torch.tensor(delta_norm)):raise ValueError('Invalid control norm')
        if original_norm==0 and delta_norm!=0:raise ValueError('Cannot rescale zero update')
        scale=delta_norm/original_norm if original_norm else 1.
    record={'mode':mode,'alpha':alpha,'selected_unique_elements':sum(p.numel() for n,p,aliases in entries if n in selected),'selected_delta_norm_before_control':original_norm,'control_scale':scale,'changed_unique_elements':0,'actual_replacement_norm':0.,'parameter_hashes_before':hashes,'parameter_hashes_during':{},'restoration_verified':False}
    try:
        with torch.no_grad():
            for n,p in touched:
                old=backups[n];x=b[n].detach().cpu()
                if n not in selected:target=x
                elif mode=='restore':
                    removal=(1-alpha)*scale
                    target=old if removal==0 else (x if removal==1 else old-removal*(old-x))
                else:target=old if alpha==1 and scale==1 else (x if alpha==0 else x+alpha*scale*(old-x))
                p.copy_(target.to(p.device))
                new=p.detach().cpu();record['changed_unique_elements']+=int((new!=old).sum());record['actual_replacement_norm']+=float((new.double()-old.double()).square().sum());record['parameter_hashes_during'][n]=digest_tensor(new)
        record['actual_replacement_norm']=record['actual_replacement_norm']**.5
        yield record
    finally:
        with torch.no_grad():
            for n,p in touched:p.copy_(backups[n].to(p.device))
        record['parameter_hashes_after']={n:digest_tensor(p) for n,p in touched}
        record['restoration_verified']=record['parameter_hashes_after']==hashes
        if not record['restoration_verified']:raise RuntimeError('Parameter rollback failed')
