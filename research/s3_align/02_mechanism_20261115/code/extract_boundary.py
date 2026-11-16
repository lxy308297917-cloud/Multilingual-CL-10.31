from pathlib import Path
import json,os,time,argparse,hashlib,subprocess
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from safetensors.torch import save_file
R=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['Base','A'],required=True);a=ap.parse_args()
 p=json.loads((R/'protocol.json').read_text());source=R/'inputs/representation.jsonl'
 assert hashlib.sha256(source.read_bytes()).hexdigest()==p['input_hashes'][source.name]
 rows=[json.loads(x) for x in source.read_text().splitlines()];out=R/'states'/f'{a.model}.safetensors';assert not out.exists()
 uuid=os.environ['CUDA_VISIBLE_DEVICES'];assert uuid.startswith('GPU-')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert not any(uuid in s for s in apps.splitlines())
 torch.set_num_threads(4);start=time.monotonic()
 model=AutoModelForCausalLM.from_pretrained(p['models']['Base'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
 if a.model!='Base':
  weight=Path(p['models'][a.model])/'adapter_model.safetensors'
  assert hashlib.sha256(weight.read_bytes()).hexdigest()==p['identities'][a.model]['sha256']
  model=PeftModel.from_pretrained(model,p['models'][a.model],is_trainable=False).merge_and_unload()
 model=model.cuda().eval();model.requires_grad_(False);states={};checks=[];timings=[]
 def forward(batch):
  n=max(len(r['input_ids']) for r in batch);ids=torch.zeros(len(batch),n,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
  for i,r in enumerate(batch):ids[i,:len(r['input_ids'])]=torch.tensor(r['input_ids'],device='cuda');mask[i,:len(r['input_ids'])]=1
  values={};handles=[]
  for layer in p['layers']:
   def hook(m,args,out,l=layer):
    h=out[0] if isinstance(out,tuple) else out
    values[f'L{l}_boundary']=torch.stack([h[i,len(r['input_ids'])-1].float() for i,r in enumerate(batch)]).cpu()
    if l==18:values['L18_body']=torch.stack([h[i,r['body_positions']].float().mean(0) for i,r in enumerate(batch)]).cpu()
   handles.append(model.model.layers[layer-1].register_forward_hook(hook))
  try:
   with torch.inference_mode():model.model(input_ids=ids,attention_mask=mask,use_cache=False)
  finally:
   for h in handles:h.remove()
  return values
 for bs in [1,4,8]:
  torch.cuda.synchronize();t=time.monotonic()
  for j in range(0,16,bs):forward([r['encoded']['en'] for r in rows[j:j+bs]])
  torch.cuda.synchronize();timings.append(dict(batch=bs,samples_per_second=16/(time.monotonic()-t),peak_bytes=torch.cuda.max_memory_allocated()))
 bs=max(timings,key=lambda x:x['samples_per_second'])['batch']
 for l in p['languages']:
  single=forward([rows[0]['encoded'][l]]);multi=forward([r['encoded'][l] for r in rows[:bs]])
  rel=max(float((single[k][0]-multi[k][0]).norm()/single[k][0].norm()) for k in single);assert rel<.02,(l,rel)
  checks.append(dict(lang=l,max_batch_relative_difference=rel));parts={}
  for i in range(0,len(rows),bs):
   values=forward([r['encoded'][l] for r in rows[i:i+bs]])
   for k,v in values.items():parts.setdefault(k,[]).append(v)
  for k,v in parts.items():states[f'{l}.{k}']=torch.cat(v).contiguous()
  print(json.dumps(dict(model=a.model,language=l,seconds=time.monotonic()-start)),flush=True)
 assert all(torch.isfinite(x).all() for x in states.values());save_file(states,str(out))
 (R/'states'/f'{a.model}.json').write_text(json.dumps(dict(status='COMPLETE',seconds=time.monotonic()-start,batch=bs,throughput=timings,checks=checks,gpu=uuid,pid=os.getpid(),peak_bytes=torch.cuda.max_memory_allocated(),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),torch=torch.__version__),indent=2))
if __name__=='__main__':main()
