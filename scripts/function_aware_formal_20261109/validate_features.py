import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from losses import collect_states,weighted_loss
def main():
 torch.set_num_threads(2);torch.manual_seed(1109)
 c=Qwen3Config(vocab_size=97,hidden_size=16,intermediate_size=32,num_hidden_layers=36,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64)
 c._attn_implementation='eager';model=Qwen3ForCausalLM(c).eval()
 row={'input_ids':[3,7,11,13,17,19],'positions':[2,3,4],'answer_prediction_slots':[0,2]}
 states=collect_states(model,row);assert states.shape==(2,72,16)
 ids=torch.tensor([row['input_ids']])
 reference=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),output_hidden_states=True,use_cache=False).hidden_states
 # HF final layer state is normalized, so use explicit block hooks for the raw36 oracle.
 blocks=[];hooks=[b.register_forward_hook(lambda m,i,o:blocks.append((o[0] if isinstance(o,tuple) else o)[0,[2,4],:])) for b in model.model.layers]
 model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
 for h in hooks:h.remove()
 assert torch.equal(states[:,1::2],torch.stack(blocks,dim=1))
 ref=(states.detach()+torch.randn_like(states)*.01).requires_grad_(False)
 actual=weighted_loss(states,ref,[1.]*6,.7)
 independent=((states-ref).square().sum((0,2))/ref.square().sum((0,2)).clamp_min(1e-12)).mean()/.7
 assert torch.equal(actual,independent)
 ga=torch.autograd.grad(actual,model.model.layers[17].self_attn.q_proj.weight,retain_graph=True)[0]
 gb=torch.autograd.grad(independent,model.model.layers[17].self_attn.q_proj.weight,retain_graph=True)[0]
 assert torch.equal(ga,gb)
 w=[.5,.6,.7,1.2,1.4,1.6];weighted=weighted_loss(states,ref,w,.7)
 per=(states-ref).square().sum((0,2))/ref.square().sum((0,2)).clamp_min(1e-12)
 grouped=torch.stack([per[24*(g//2)+g%2:24*(g//2)+24:2].mean() for g in range(6)])
 assert torch.allclose(weighted,(grouped*torch.tensor(w)).mean()/.7,rtol=1e-6)
 weighted.backward();assert all(b.self_attn.q_proj.weight.grad is not None and b.mlp.down_proj.weight.grad is not None for b in model.model.layers)
 print('PASS 72 residual feature sites; odd sites exactly match prior36 block outputs; uniform loss and gradients exact independent oracle; all six coefficients map correctly; gradients reach every layer and both module types.')
if __name__=='__main__':main()
