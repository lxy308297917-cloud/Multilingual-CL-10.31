"""Common 72-site representation objective; only the six group coefficients differ."""
import json, os
from pathlib import Path
import torch
from safetensors.torch import load_file
from logit_kd import old_logit_kd_objective
import trainer_core
trainer_core.torch=torch
import time
trainer_core.time=time
TEACHER=Path('/path/to/storage/datasets/crosslayer-formal-20261109/teacher-v1')
_CACHE=None
_WEIGHTS=None

def collect_states(model,row):
    raw=model.get_base_model() if hasattr(model,'get_base_model') else model
    ids=torch.tensor([row['input_ids']],device=next(raw.parameters()).device)
    positions=[row['positions'][slot] for slot in row['answer_prediction_slots']]
    states={}; hooks=[]
    for layer,block in enumerate(raw.model.layers):
        def pre(module,inputs,layer=layer): states[2*layer]=inputs[0][0,positions,:]
        def post(module,inputs,output,layer=layer):
            output=output[0] if isinstance(output,tuple) else output
            states[2*layer+1]=output[0,positions,:]
        hooks.append(block.post_attention_layernorm.register_forward_pre_hook(pre))
        hooks.append(block.register_forward_hook(post))
    try: raw.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    finally:
        for hook in hooks: hook.remove()
    assert len(states)==2*len(raw.model.layers)
    return torch.stack([states[i] for i in range(len(states))],dim=1).float()

def weighted_loss(current,reference,weights,normalizer):
    reference=reference.detach().to(current)
    assert current.shape==reference.shape and current.shape[1]==72
    per_site=(current-reference).square().sum((0,2))/reference.square().sum((0,2)).clamp_min(1e-12)
    group=torch.tensor([weights[2*(i//24)+i%2] for i in range(72)],device=current.device,dtype=current.dtype)
    assert abs(float(group.mean())-1)<1e-6
    return (per_site*group).mean()/normalizer

def old_objective(model,row,mode,reference=None,normalizer=1.):
    global _CACHE,_WEIGHTS
    assert mode=='full_feature'
    method=os.environ['FA_METHOD']
    if method=='kd':
        if _CACHE is None:
            _CACHE=load_file(str(TEACHER/'teacher_log_probs.safetensors'))
        scales=json.loads((TEACHER/'manifest.json').read_text())['normalizers']
        return old_logit_kd_objective(model,row,_CACHE[row['id']],scales[row['capability']])
    if _WEIGHTS is None:
        _WEIGHTS=[1.]*6 if method=='full' else json.loads(Path(os.environ['FA_WEIGHTS']).read_text())[method]
        assert len(_WEIGHTS)==6 and abs(sum(_WEIGHTS)-6)<1e-7 and min(_WEIGHTS)>=.5-1e-7 and max(_WEIGHTS)<=2+1e-7
    return weighted_loss(collect_states(model,row),reference,_WEIGHTS,normalizer)

trainer_core.old_objective=old_objective
trainer_class=trainer_core.trainer_class
