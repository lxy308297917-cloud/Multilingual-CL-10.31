"""One combined loss/backward per target microbatch for ZeRO2 CPU offload."""
import torch

def raw_model(model):
 while hasattr(model,'module'):model=model.module
 assert not hasattr(model,'peft_config'),'FFT only, no PEFT adapter'
 return model

def collect_raw_layers(model,row):
 """Inputs to layers1..35 plus final norm = original raw block outputs0..35.

 Store full tensors in hooks and slice AFTER forward, preserving the graph when
 checkpointing is active. Never capture a detached slice inside a checkpoint.
 """
 raw=raw_model(model);states={};hooks=[]
 ids=torch.tensor([row['input_ids']],device=next(raw.parameters()).device,dtype=torch.long)
 positions=[row['positions'][s] for s in row['answer_prediction_slots']]
 n=len(raw.model.layers)
 for index,block in enumerate(raw.model.layers[1:]):
  def pre(module,inputs,index=index):states[index]=inputs[0]
  hooks.append(block.register_forward_pre_hook(pre))
 def last(module,inputs):states[n-1]=inputs[0]
 hooks.append(raw.model.norm.register_forward_pre_hook(last))
 try:raw.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
 finally:
  for hook in hooks:hook.remove()
 assert len(states)==n
 current=torch.stack([states[i][0,positions,:] for i in range(n)],dim=1).float()
 assert current.requires_grad or not torch.is_grad_enabled()
 return current

def representation_loss(model,row,reference,normalizer):
 current=collect_raw_layers(model,row);ref=reference.detach().to(current)
 assert current.shape==ref.shape and normalizer>0
 return ((current-ref).square().sum((0,2))/ref.square().sum((0,2)).clamp_min(1e-12)).mean()/normalizer

def auxiliary_coefficient(ga,model_accepts_loss_kwargs,num_items_in_batch):
 assert ga==16
 # Eight old rows per optimizer step, matching P2 micro2/GA8 exposure.
 # HF either receives globally-normalized target CE, or divides both terms later.
 return 2/ga if model_accepts_loss_kwargs and num_items_in_batch is not None else 2.

def trainer_class(upstream):
 class RepresentationFFTTrainer(upstream):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs)
   self.old_trace=[];self.backend_backward_calls=0;self.micro=-1;self.previous_step=None
   original=self.accelerator.backward
   def backward(loss,**kw):
    self.backend_backward_calls+=1
    assert kw.get('scale_wrt_gas') is False,'ZeRO2 must not divide the already scaled loss twice'
    return original(loss,**kw)
   self.accelerator.backward=backward
  def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
   target,outputs=super().compute_loss(model,inputs,return_outputs=True,num_items_in_batch=num_items_in_batch)
   if not model.training or self.protection_mode=='none':return (target,outputs) if return_outputs else target
   if self.previous_step!=self.state.global_step:self.previous_step=self.state.global_step;self.micro=0
   micro=self.micro;self.micro+=1
   assert 0<=micro<16
   if micro%2:return (target,outputs) if return_outputs else target
   index=self.state.global_step*8+micro//2;row=self.old_rows[index%128];assert row['split']=='build'
   # Qwen3 FFT has no adapter dropout, attention_dropout=0. Train/eval function
   # agrees; keeping train mode enables non-reentrant activation checkpointing.
   old=representation_loss(model,row,self.references[row['id']],self.normalizers[row['capability']])
   assert torch.isfinite(old)
   factor=auxiliary_coefficient(self.args.gradient_accumulation_steps,self.model_accepts_loss_kwargs,num_items_in_batch)
   total=target+factor*old
   self.old_trace.append(dict(step=self.state.global_step+1,micro=micro,id=row['id'],split=row['split'],capability=row['capability'],input_tokens=len(row['input_ids']),reference_prefix_tokens=len(row['input_ids'])-row['prefix_length'],old_objective=float(old.detach()),old_coefficient_inside_compute_loss=factor,target_global_token_normalization=self.model_accepts_loss_kwargs and num_items_in_batch is not None,protected_positions=len(row['answer_prediction_slots']),layers=36))
   return (total,outputs) if return_outputs else total
 return RepresentationFFTTrainer
