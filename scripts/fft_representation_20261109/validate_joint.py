"""CPU correctness: raw36 features, non-reentrant gradients, GA scaling."""
import copy
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from joint_loss import collect_raw_layers,representation_loss,auxiliary_coefficient

def main():
 torch.set_num_threads(2);torch.manual_seed(1109)
 c=Qwen3Config(vocab_size=97,hidden_size=16,intermediate_size=32,num_hidden_layers=36,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64)
 c._attn_implementation='eager';model=Qwen3ForCausalLM(c).train();row=dict(input_ids=[3,7,11,13,17,19],positions=[2,3,4],answer_prediction_slots=[0,2]);ids=torch.tensor([row['input_ids']])
 plain=copy.deepcopy(model);model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 states=collect_raw_layers(model,row);oracle=[]
 handles=[b.register_forward_hook(lambda m,i,o:oracle.append((o[0] if isinstance(o,tuple) else o)[0,[2,4],:])) for b in plain.model.layers]
 plain.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
 for h in handles:h.remove()
 expected=torch.stack(oracle,dim=1).float();assert torch.equal(states,expected)
 teacher=expected.detach()+torch.randn_like(expected)*.01
 a=representation_loss(model,row,teacher,.7);b=((expected-teacher).square().sum((0,2))/teacher.square().sum((0,2)).clamp_min(1e-12)).mean()/.7
 a.backward();b.backward()
 for (n,p),(m,q) in zip(model.named_parameters(),plain.named_parameters()):
  assert n==m
  if p.grad is not None:assert q.grad is not None and torch.allclose(p.grad,q.grad,rtol=2e-5,atol=2e-6),(n,'checkpoint gradient mismatch')
 assert model.model.layers[17].self_attn.q_proj.weight.grad.abs().sum()>0 and model.model.embed_tokens.weight.grad.abs().sum()>0
 # Independent scalar GA oracle: old losses occur8 times, exactly weight1 total.
 for accepts,num_items,downstream_divisor in [(True,100,1),(False,None,16),(True,None,16)]:
  coefficient=auxiliary_coefficient(16,accepts,num_items)
  assert abs(8*coefficient/downstream_divisor-1)<1e-12
 print('PASS raw36 sites exactly match original block outputs; non-reentrant checkpointing preserves representation gradients; full-model gradients reach embeddings and decoder weights; eight-old-row GA16 scaling equals frozen lambda1 in all HF normalization branches. No benchmark scores used.')
if __name__=='__main__':main()
