"""Three deliberately simple auxiliary baselines, four rotations, T=1."""
import torch

def content_average(label_logits, orders):
    probs=label_logits.float().softmax(-1)
    inv=torch.argsort(torch.as_tensor(orders,device=probs.device),dim=1)
    return probs.gather(1,inv).mean(0)

def auxiliary_loss(mode,label_logits,orders,gold,teacher_content,teacher_marginal):
    if mode=='replay':
        gold_positions=torch.tensor([order.index(gold) for order in orders],device=label_logits.device)
        return torch.nn.functional.cross_entropy(label_logits.float(),gold_positions)
    student=(label_logits.float().softmax(-1).mean(0) if mode=='marginal' else content_average(label_logits,orders))
    target=torch.as_tensor(teacher_marginal if mode=='marginal' else teacher_content,device=student.device,dtype=student.dtype)
    assert mode in ['marginal','content']
    return torch.sum(torch.where(target>0,target*(target.clamp_min(1e-30).log()-student.clamp_min(1e-30).log()),torch.zeros_like(target)))
