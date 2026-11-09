"""T=1 dense Base-to-student distillation on the existing protected answer positions.

CPU-prepared strong-control primitive. This file does not start a training job.
"""
import torch
import torch.nn.functional as F

def teacher_to_student_kl(student_logits,teacher_log_probs,normalizer=1.0):
 assert student_logits.ndim==2 and student_logits.shape==teacher_log_probs.shape
 assert student_logits.shape[0]>0 and normalizer>0
 teacher=teacher_log_probs.detach().to(device=student_logits.device,dtype=student_logits.dtype)
 assert torch.isfinite(teacher).all()
 # Mean over protected prediction positions, SUM over the vocabulary (not vice versa).
 return F.kl_div(F.log_softmax(student_logits,dim=-1),teacher,reduction='batchmean',log_target=True)/normalizer

def old_logit_kd_objective(model,row,teacher_log_probs,normalizer=1.0):
 raw=model.get_base_model()if hasattr(model,'get_base_model')else model
 ids=torch.tensor([row['input_ids']],dtype=torch.long,device=next(raw.parameters()).device)
 positions=[row['positions'][slot]for slot in row['answer_prediction_slots']]
 assert 0<len(positions)<=4 and all(p>=row['prefix_length']-1 and p<ids.shape[1]-1 for p in positions)
 hidden=raw.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
 logits=raw.lm_head(hidden[0,positions,:]).float()
 return teacher_to_student_kl(logits,teacher_log_probs,normalizer)

def validate():
 from transformers import Qwen3Config,Qwen3ForCausalLM
 torch.set_num_threads(2);torch.manual_seed(1109)
 # KL orientation, normalization and gradient have an independent closed-form oracle.
 logits=torch.randn(3,7,dtype=torch.float64,requires_grad=True)
 teacher=F.log_softmax(torch.randn(3,7,dtype=torch.float64),dim=-1)
 value=teacher_to_student_kl(logits,teacher,.37)
 oracle=(teacher.exp()*(teacher-F.log_softmax(logits,-1))).sum()/3/.37
 assert torch.allclose(value,oracle,atol=1e-12,rtol=1e-12)
 gradient=torch.autograd.grad(value,logits)[0]
 expected=(logits.softmax(-1)-teacher.exp())/3/.37
 assert torch.allclose(gradient,expected,atol=1e-12,rtol=1e-12)
 cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128)
 cfg._attn_implementation='eager';model=Qwen3ForCausalLM(cfg).eval()
 row=dict(input_ids=[3,5,7,11,13,17,19,23],prefix_length=4,positions=[2,3,5],answer_prediction_slots=[1,2])
 ids=torch.tensor([row['input_ids']]);positions=[3,5]
 with torch.no_grad():teacher=F.log_softmax(model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,positions].float(),dim=-1)
 zero=old_logit_kd_objective(model,row,teacher)
 assert abs(float(zero))<1e-6
 with torch.no_grad():model.model.layers[1].mlp.down_proj.weight.add_(torch.randn_like(model.model.layers[1].mlp.down_proj.weight)*.03)
 x=old_logit_kd_objective(model,row,teacher,.25)
 full=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,positions].float()
 y=teacher_to_student_kl(full,teacher,.25)
 assert torch.allclose(x,y,atol=1e-7,rtol=1e-6)and float(x)>0
 x.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
 assert not teacher.requires_grad
 print('PASS KL(Base||student) vocabulary-sum/position-mean and analytical gradient; identical teacher near-zero; exact protected-slot alignment against full HF forward; teacher detached. CPU only, KD training NOT_RUN.')
if __name__=='__main__':validate()
