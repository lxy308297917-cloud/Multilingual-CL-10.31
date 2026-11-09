"""Fixed build-only CE calibration is part of the algorithm, never a score gate."""
import argparse,datetime,hashlib,json,os,re,sys,time
from pathlib import Path
import numpy as np
R=Path('/path/to/project')
C=R/'configs/function_aware_formal_20261109'
D=Path('/path/to/storage/datasets/function-aware-formal-20261109/calibration-v1')
L=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109')
P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
GROUPS=[b+'_'+m for b in ['early','middle','late'] for m in ['attention','mlp']]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def group(name):
 layer=int(re.search(r'\.layers\.(\d+)\.',name)[1]);return 2*(layer//12)+(0 if '.self_attn.' in name else 1)
def project(values):
 v=np.asarray(values,dtype=float);lo=float(v.min()-2);hi=float(v.max()+2)
 for _ in range(100):
  mid=(lo+hi)/2
  if np.clip(v-mid,.5,2).sum()>6:lo=mid
  else:hi=mid
 w=np.clip(v-(lo+hi)/2,.5,2);assert abs(w.sum()-6)<1e-9
 return w.tolist()
def finish(lang,out,scores):
 rows=[json.loads(s) for s in (D/(lang+'.jsonl')).open()];baseline=scores['all'];risks=[];details=[]
 for g in range(6):
  delta={}
  for task in ['old_math','old_instruction',lang+'_reading',lang+'_summary']:
   ids=[x['id'] for x in rows if x['task']==task]
   delta[task]=float(np.mean([(scores[f'g{g}_uniform'][i]-scores[f'g{g}_group'][i])/max(baseline[i],.1) for i in ids]))
  old=(delta['old_math']+delta['old_instruction'])/2
  target=-(delta[lang+'_reading']+delta[lang+'_summary'])/2
  risk=old-max(0,target);risks.append(risk);details.append(dict(group=GROUPS[g],relative_recoveries=delta,old_recovery=old,target_cost=target,risk=risk))
 positive=np.maximum(risks,0);weights=[1.]*6 if positive.sum()==0 else project(.5+3*positive/positive.sum())
 perm=np.random.default_rng(61415).permutation(6).tolist();random=[weights[i] for i in perm]
 result=dict(status='WEIGHTS_FROZEN_NO_PERFORMANCE_GATE',language=lang,groups=GROUPS,function=weights,random=random,random_permutation=perm,sum_weight=6,bounds=[.5,2],details=details,calibration_manifest_sha256=sha(D/'manifest.json'),calibration_raw_sha256=sha(out/'predictions.jsonl'),algorithm='relative CE recovery - positive target CE cost; positive-score allocation; Euclidean capped-simplex projection',source='existing independently trained per-language adapter; all build rows; no formal outcomes')
 target=C/(lang+'_weights.json');assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
 (out/'DONE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
def sw_reuse(out):
 # Same P2 batch2 SW calibration adapter; reuse exact previously computed logits/CE.
 cfg=json.loads((R/'configs/functional_risk_p3_20261109/interventions.json').read_text())
 old=Path('/path/to/logs/qwen3-multilingual/functional-risk-p3-20261109/ce-mc-v3')
 rows=[json.loads(s) for s in (D/'sw.jsonl').open()];ids={x['id'] for x in rows};scores={};raw=[]
 names={'all':'all_layer'}
 for g,n in enumerate(GROUPS):
  for kind in ['group','uniform']:names[f'g{g}_{kind}']='anatomical_'+n+'_'+kind
 for v,source in names.items():
  path=old/source/'predictions.jsonl';done=json.loads((path.parent/'DONE.json').read_text());assert sha(path)==done['predictions_sha256']
  selected=[x for x in map(json.loads,path.open()) if 'nll' in x and x['split']=='build' and x['id'] in ids]
  assert len(selected)==224 and {x['id'] for x in selected}==ids
  scores[v]={x['id']:x['nll'] for x in selected};raw.extend(dict(x,variant=v,reused=True) for x in selected)
 (out/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in raw));finish('sw',out,scores)
def gpu_worker(lang,out):
 import torch
 from transformers import AutoModelForCausalLM
 from peft import PeftModel
 from safetensors.torch import load_file
 torch.set_num_threads(4)
 audit=json.loads((R/'research/function_aware_formal_20261109/reuse_audit.json').read_text())[lang]
 source=Path(audit['original_adapter']);assert sha(source/'adapter_model.safetensors')==audit['original_adapter_sha256']
 state=load_file(str(source/'adapter_model.safetensors'));keys=[k for k in state if k.endswith('lora_A.weight')]
 energy=np.zeros(6)
 for k in keys:
  a=state[k].double();b=state[k.replace('lora_A','lora_B')].double();energy[group(k)]+=float(((b.T@b)*(a@a.T).T).sum()*4)
 total=float(energy.sum());beta=np.sqrt((total-(1-.75**2)*energy)/total)
 model=AutoModelForCausalLM.from_pretrained('/path/to/models/Qwen3-4B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
 model=PeftModel.from_pretrained(model,source,is_trainable=False).eval().requires_grad_(False);raw=model.get_base_model()
 params={n:p for n,p in model.named_parameters() if '.lora_B.' in n};assert len(params)==252
 initial={n:p.detach().clone() for n,p in params.items()};rows=[json.loads(s) for s in (D/(lang+'.jsonl')).open()];assert len(rows)==224
 variants=[('all',None,None)]+[(f'g{g}_{kind}',g,kind) for g in range(6) for kind in ['group','uniform']]
 scores={}
 with (out/'predictions.jsonl').open('x') as stream,torch.inference_mode():
  for name,g,kind in variants:
   for n,p in params.items():
    factor=1. if g is None else float(beta[g]) if kind=='uniform' else .75 if group(n)==g else 1.
    p.copy_(initial[n]*factor)
   values={};started=time.monotonic()
   for i,row in enumerate(rows):
    ids=torch.tensor([row['input_ids']],device='cuda');prefix=row['prefix_length'];h=raw.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
    logits=raw.lm_head(h[:,prefix-1:-1,:]).float();value=float(torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,prefix:].reshape(-1)))
    values[row['id']]=value;stream.write(json.dumps(dict(variant=name,id=row['id'],task=row['task'],nll=value))+'\n')
    del ids,h,logits
   stream.flush();scores[name]=values;print(json.dumps(dict(variant=name,rows=len(rows),elapsed_seconds=time.monotonic()-started)),flush=True)
 (out/'norm_controls.json').write_text(json.dumps(dict(group_energy=energy.tolist(),total_energy=total,uniform_beta=beta.tolist(),attenuation=.75),indent=2)+'\n')
 finish(lang,out,scores)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--language',choices=['sw','ar','ru'],required=True);ap.add_argument('--worker',action='store_true');ap.add_argument('--gpu-uuid');a=ap.parse_args();out=L/(a.language+'_calibration')
 if a.worker:
  assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['AYA_EXPECTED_GPU_UUID']==a.gpu_uuid
  gpu_worker(a.language,out);return
 out.mkdir(parents=True,exist_ok=False)
 if a.language=='sw':sw_reuse(out);return
 sys.path.insert(0,str(R/'scripts/aya_4b_multilang_20261106'));from gpu_slot import run_job
 def save(status,**kw):(out/'state.json').write_text(json.dumps(dict(status=status,updated_at=datetime.datetime.now().astimezone().isoformat(),**kw),indent=2)+'\n')
 env=dict(os.environ,HF_HOME='/path/to/storage/hf-cache-qwen3',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4')
 try:run_job(lambda uuid:[P,str(Path(__file__).resolve()),'--language',a.language,'--worker','--gpu-uuid',uuid],env,out/'worker.log',save);save('COMPLETE')
 except Exception as exc:save('FAILED',error=repr(exc));raise
if __name__=='__main__':main()
