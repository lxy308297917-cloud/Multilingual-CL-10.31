from common import *
import argparse,torch,time
from transformers import AutoModelForCausalLM
from peft import PeftModel
from safetensors.torch import load_file
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);a=ap.parse_args()
 torch.set_num_threads(4);start=time.monotonic();done=read(a.checkpoint/'TRAINING_DONE.json')
 assert sha(a.checkpoint/'adapter_model.safetensors')==done['weights_sha256']['adapter_model.safetensors']
 base=AutoModelForCausalLM.from_pretrained(str(BASE),torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',local_files_only=True)
 model=PeftModel.from_pretrained(base,str(a.checkpoint)).eval();pool=rows(R/'inputs/alignment.jsonl');states=load_file(str(PREV/'states/Base.safetensors'))
 checks=[]
 for i in [0,1]:
  row=pool[i]['encoded']['en'];ids=torch.tensor([row['input_ids']],device='cuda');saved=[]
  def hook(module,args,out):saved.append((out[0] if isinstance(out,tuple) else out)[0,row['body_positions']].float().mean(0))
  handle=base.model.layers[17].register_forward_hook(hook)
  with torch.inference_mode(),model.disable_adapter():output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
  handle.remove();expected=states['en.L18_body'][i].cuda().float();rel=float((saved[0]-expected).norm()/expected.norm())
  assert rel<.03 and torch.isfinite(output.logits).all(),rel
  with torch.inference_mode():out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
  assert torch.isfinite(out.logits).all()
  checks.append(dict(id=pool[i]['id'],frozen_base_cache_relative_error=rel,adapter_logit_delta=float((out.logits-output.logits).float().norm())))
 write(a.checkpoint/'RELOAD_VERIFIED.json',dict(at=now(),weights_sha256=done['weights_sha256'],checks=checks,seconds=time.monotonic()-start,teacher_gate=.03,teacher_gate_explanation='BF16 kernel/cache tolerance fixed before check; verifies underlying frozen Base identity on2inputs, not exhaustive numerical equivalence'))
 print(json.dumps(checks))
if __name__=='__main__':main()
