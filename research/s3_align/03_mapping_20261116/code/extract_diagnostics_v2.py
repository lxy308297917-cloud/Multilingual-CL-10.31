from common import *
from jsonl_safe import rows
import argparse,torch,time,os
from transformers import AutoModelForCausalLM
from peft import PeftModel
from safetensors.torch import save_file
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['Base','A','B','E'],required=True);a=ap.parse_args()
 torch.set_num_threads(4);protocol=read(PREV/'protocol.json');path=protocol['models'][a.model]
 if a.model!='Base':assert sha(Path(path)/'adapter_model.safetensors')==protocol['identities'][a.model]['sha256']
 for plan in ['MATH_DATA_FROZEN.json','RELATION_DATA_FROZEN.json']:
  for p,d in read(R/'records'/plan)['hashes'].items():assert sha(p)==d,p
 out=R/'assets/diagnostic_states_v2'/a.model;out.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();base=AutoModelForCausalLM.from_pretrained(str(BASE),torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',local_files_only=True)
 model=base if a.model=='Base' else PeftModel.from_pretrained(base,path);model.eval();torch.cuda.reset_peak_memory_stats()
 layers=base.model.layers;math=rows(R/'inputs/math256.jsonl');tensors={};records=[]
 for i,row in enumerate(math):
  ids=torch.tensor([row['input_ids']],device='cuda');positions=row['positions'];saved={};hooks=[]
  for layer in [12,18,24]:
   def hook(mod,args,out,layer=layer):saved[layer]=(out[0] if isinstance(out,tuple) else out)[0,positions].float()
   hooks.append(layers[layer-1].register_forward_hook(hook))
  try:
   with torch.inference_mode():
    hidden=base.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state[0,positions]
    logits=base.lm_head(hidden).float();logp=torch.log_softmax(logits,dim=-1);targets=torch.tensor(row['target_ids'],device='cuda')
    chosen=logp.gather(1,targets[:,None]).flatten();pred=logp.argmax(1)
  finally:
   for h in hooks:h.remove()
  for layer,h in saved.items():tensors[f"math.{row['id']}.L{layer}"]=h.cpu().contiguous()
  records.append(dict(id=row['id'],positions=positions,target_ids=row['target_ids'],target_log_probs=chosen.cpu().tolist(),predicted_token_ids=pred.cpu().tolist(),prompt_tokens=row['prompt_tokens'],answer_tokens=row['answer_tokens'],reference_steps=row['reference_steps']))
  if (i+1)%64==0:print(a.model,'math',i+1,flush=True)
 save_file(tensors,str(out/'math_states.safetensors'));jsonl(out/'math_reference_probabilities.jsonl',records);del tensors
 if a.model in ['Base','A']:
  relation=rows(R/'inputs/relations64.jsonl');tensors={};batchcheck=[]
  for i,row in enumerate(relation):
   for key,enc in row['encoded'].items():
    ids=torch.tensor([enc['input_ids']],device='cuda');saved=[]
    def hook(mod,args,out):
     h=out[0] if isinstance(out,tuple) else out;saved.extend([h[0,enc['body_positions']].float().mean(0),h[0,-1].float()])
    handle=layers[17].register_forward_hook(hook)
    try:
     with torch.inference_mode():base.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    finally:handle.remove()
    assert len(saved)==2
    for pos,h in zip(['body','boundary'],saved):tensors[f"{row['id']}.{key}.{pos}"]=h.cpu().contiguous()
   if (i+1)%16==0:print(a.model,'relations',i+1,flush=True)
  save_file(tensors,str(out/'relation_states.safetensors'))
 write(out/'DONE.json',dict(at=now(),model=a.model,model_path=path,adapter_sha256=None if a.model=='Base' else sha(Path(path)/'adapter_model.safetensors'),math_n=256,relation_n=64 if a.model in ['Base','A'] else 0,layers=[12,18,24],batch_size=1,seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],input_hashes={str(p):sha(p) for p in [R/'inputs/math256.jsonl',R/'inputs/relations64.jsonl']},code_sha256=sha(__file__),logprob_precision='FP32 log_softmax of BF16 model outputs; teacher-forcing diagnostic only, not formal task scorer'))
if __name__=='__main__':main()
