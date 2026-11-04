"""Full-vocabulary gold-answer-token SFT replay. No candidate restriction."""
import torch
from losses import auxiliary_loss

def full_vocabulary_replay(logits,gold_tokens):
    targets=torch.as_tensor(gold_tokens,device=logits.device,dtype=torch.long)
    return torch.nn.functional.cross_entropy(logits.float(),targets)
