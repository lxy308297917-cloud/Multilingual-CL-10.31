"""Scores with explicit continuation and shifted assistant-token masking."""
import torch
import torch.nn.functional as F

def summarize_options(scores,gold):
 s=torch.as_tensor(scores,dtype=torch.float64);assert s.shape==(4,) and torch.isfinite(s).all();assert gold in range(4)
 wrong=s.clone();wrong[gold]=-torch.inf
 return {'predicted':int(s.argmax()),'correct':bool(int(s.argmax())==gold),'correct_answer_sequence_log_probability':float(s[gold]),'correct_answer_4option_log_probability':float(s.log_softmax(0)[gold]),'gold_minus_best_wrong_margin':float(s[gold]-wrong.max())}

def shifted_assistant_nll(logits,labels):
 scores=logits.float()[:,:-1,:];targets=labels[:,1:];keep=targets!=-100;n=int(keep.sum());assert n>0
 loss=F.cross_entropy(scores.reshape(-1,scores.shape[-1]),targets.reshape(-1),ignore_index=-100,reduction='sum');assert torch.isfinite(loss)
 return float(loss),n

def option_log_probabilities(model,context,continuations,device):
 assert context and len(continuations)==4 and all(continuations)
 # Common continuation whitespace contributes equally, but is kept in raw LP.
 prefix=continuations[0][:-1]
 if all(c[:-1]==prefix for c in continuations):
  ids=torch.tensor([context+prefix],device=device)
  logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(prefix)+1).logits[0].float().log_softmax(-1)
  common=sum(logits[i,t] for i,t in enumerate(prefix)) if prefix else logits.new_tensor(0.)
  return [float(common+logits[-1,c[-1]]) for c in continuations]
 scores=[]
 for c in continuations:
  ids=torch.tensor([(context+c)[:-1]],device=device);logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(c)).logits[0].float().log_softmax(-1)
  scores.append(float(logits[torch.arange(len(c),device=device),torch.tensor(c,device=device)].sum()))
 return scores
