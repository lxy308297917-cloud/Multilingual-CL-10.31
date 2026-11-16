"""Historical loss formulas; training data and model execution remain external."""
import contextlib
import torch
import torch.nn.functional as F

def raw_model(model):
    return model.get_base_model() if hasattr(model, 'get_base_model') else model

def old_objective(model, row, mode, reference=None, normalizer=1.0):
    """Old forward uses eval mode but retains adapter gradients; caller restores mode."""
    raw = raw_model(model)
    ids = torch.tensor([row['input_ids']], dtype=torch.long, device=next(raw.parameters()).device)
    if mode == 'ce_replay':
        hidden = raw.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        p = row['prefix_length']
        logits = raw.lm_head(hidden[:, p - 1:-1, :]).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, p:].reshape(-1))
    assert mode == 'full_feature' and reference is not None and normalizer > 0
    states, hooks = [], []
    slots = row['answer_prediction_slots']
    positions = [row['positions'][slot] for slot in slots]
    for block in raw.model.layers:
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            states.append(h[0, positions, :])
        hooks.append(block.register_forward_hook(hook))
    try:
        raw.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in hooks:
            handle.remove()
    assert len(states) == len(raw.model.layers)
    current = torch.stack(states, dim=1).float()  # answer positions x layers x features
    ref = reference.to(device=current.device, dtype=torch.float32)
    assert current.shape == ref.shape
    per_layer = (current - ref).square().sum(dim=(0, 2)) / ref.square().sum(dim=(0, 2)).clamp_min(1e-12)
    return per_layer.mean() / normalizer

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
