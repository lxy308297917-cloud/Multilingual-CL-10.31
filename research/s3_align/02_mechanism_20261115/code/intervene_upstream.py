"""Exploratory activation patching with real upstream prompt/scoring cross-checks.
No model fitting, no optimizer, no use of future generated answers in source states.
"""
from pathlib import Path
import json,os,time,argparse,hashlib,subprocess
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file,save_file
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager,get_task_dict
R=Path(__file__).resolve().parents[1]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['Base','A'],required=True);a=ap.parse_args()
 p=json.loads((R/'protocol.json').read_text());rows=list(map(json.loads,(R/'inputs/intervention.jsonl').read_text().splitlines()))
 uuid=os.environ['CUDA_VISIBLE_DEVICES'];apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert not any(uuid in s for s in apps.splitlines())
 out=R/'analysis'/f'intervention_{a.model}.jsonl';assert not out.exists()
 torch.set_num_threads(4);start=time.monotonic()
 tok=AutoTokenizer.from_pretrained(p['models']['Base'],local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(p['models']['Base'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
 if a.model!='Base':
  weight=Path(p['models'][a.model])/'adapter_model.safetensors';assert hashlib.sha256(weight.read_bytes()).hexdigest()==p['identities'][a.model]['sha256']
  model=PeftModel.from_pretrained(model,p['models'][a.model],is_trainable=False).merge_and_unload()
 model=model.cuda().eval();model.requires_grad_(False)
 lm=HFLM(pretrained=model,tokenizer=tok,batch_size=1,max_length=8192,enable_thinking=False)
 tasks=get_task_dict(['belebele_eng_Latn_locked','belebele_swh_Latn_locked'],task_manager=TaskManager(include_path='/srv/storage/datasets/function-aware-formal-20261109/evaluation-v1/tasks'))
 tasks={l:tasks['belebele_'+code+'_locked'] for l,code in [('en','eng_Latn'),('sw','swh_Latn')]}
 encoded={};requests={};prompts=[]
 for i,r in enumerate(rows):
  for l in ['en','sw']:
   task=tasks[l];context=lm.apply_chat_template([{'role':'user','content':task.doc_to_text(r[l])}],add_generation_prompt=True)
   reqs=task.construct_requests(r[l],context,metadata=(task.config.task,i,1),apply_chat_template=True)
   ids=[];choices=[];prefixes=[]
   for req in reqs:
    x,y=lm._encode_pair(*req.args);assert len(y)>=1
    ids.append(x+y[:-1]);choices.append(y[-1]);prefixes.append((len(x),y[:-1]))
   assert all(x==ids[0] for x in ids) and len(ids[0])<8192
   assert all(v==prefixes[0] for v in prefixes) and ids[0]==lm.tok_encode(context)
   encoded[i,l]=(ids[0],choices,prefixes[0]);requests[i,l]=reqs
   prompts.append(dict(id=r['id'],language=l,context=context,input_ids=ids[0],continuations=[q.args[1] for q in reqs],choice_tokens=choices,common_continuation_prefix=prefixes[0]))
 write(R/'records'/f'{a.model}_actual_prompts.json',prompts)
 # Same-length replica batches, so the patched position is unambiguous and no padding is used.
 def forward(ids,choice_ids,prefix,deltas=None):
  batch=1 if deltas is None else len(deltas);x=torch.tensor(ids,device='cuda')[None].repeat(batch,1);saved={}
  def hook(m,args,output):
   h=output[0] if isinstance(output,tuple) else output
   saved['boundary']=h[:,-1].float().detach().cpu()
   if deltas is not None:
    h=h.clone();h[:,-1]=(h[:,-1].float()+deltas.to(h.device)).to(h.dtype)
    return (h,)+output[1:] if isinstance(output,tuple) else h
  handle=model.model.layers[17].register_forward_hook(hook)
  try:
   with torch.inference_mode():
    all_logprobs=torch.nn.functional.log_softmax(lm._model_call(x),dim=-1,dtype=lm.softmax_dtype)
    lp=all_logprobs[:,-1,choice_ids]
    # HFLM moves trailing newlines into the continuation. Score that common prefix too.
    prefix_start,prefix_tokens=prefix
    for j,target in enumerate(prefix_tokens):
     common=all_logprobs[:,prefix_start+j-1,target]
     lp=lp+common[:,None]
    lp=lp.float().cpu()
  finally:handle.remove()
  return lp,saved['boundary']
 base_states={};baseline={};audits=[]
 for i,r in enumerate(rows):
  for l in ['en','sw']:
   lp,h=forward(*encoded[i,l]);base_states.setdefault(l,[]).append(h[0]);baseline[i,l]=lp[0]
   if i<4:
    ref=lm.loglikelihood(requests[i,l],disable_tqdm=True);ref_lp=torch.tensor([v[0] for v in ref]);difference=float((lp[0]-ref_lp).abs().max())
    assert difference<.05 and int(lp[0].argmax())==int(ref_lp.argmax()),(a.model,i,l,difference)
    audits.append(dict(id=r['id'],language=l,max_logprob_error=difference,prediction_equal=True,reference=ref_lp.tolist(),custom=lp[0].tolist()))
  if i%32==0:print(json.dumps(dict(model=a.model,stage='source',rows=i+1,seconds=time.monotonic()-start)),flush=True)
 write(R/'records'/f'{a.model}_task_baselines.json',[dict(id=rows[i]['id'],language=l,gold=rows[i]['gold'],pred=int(lp.argmax()),logprobs=lp.tolist()) for (i,l),lp in baseline.items()])
 save_file({l:torch.stack(v).contiguous() for l,v in base_states.items()},str(R/'states'/f'{a.model}_task_boundary.safetensors'))
 base_states={l:torch.stack(v) for l,v in base_states.items()}
 bases=load_file(str(R/'analysis/Base_bases.safetensors'));own=load_file(str(R/'analysis'/f'{a.model}_bases.safetensors'))
 q=bases['L18_boundary.joint64'];old=load_file('/srv/storage/datasets/s3-bactrian-20261113/teacher-v1/teacher.safetensors')['shared']
 randoms=[bases[f'L18_boundary.random{s}'] for s in p['random_seeds']]
 source_base=load_file(str(R/'states/Base_task_boundary.safetensors')) if a.model=='A' else None
 # Throughput and numerical gate for replica batches using real MCQA contexts.
 ids,choice,prefix=encoded[0,'sw'];single,_=forward(ids,choice,prefix);timings=[]
 for bs in [1,4,8]:
  torch.cuda.synchronize();t=time.monotonic();many,_=forward(ids,choice,prefix,torch.zeros(bs,2560));torch.cuda.synchronize()
  diff=float((many-single).abs().max());timings.append(dict(batch=bs,samples_per_second=bs/(time.monotonic()-t),max_logprob_error=diff,prediction_equal=bool((many.argmax(1)==single.argmax(1)).all()),peak_bytes=torch.cuda.max_memory_allocated()))
 valid=[x for x in timings if x['max_logprob_error']<.05 and x['prediction_equal']];assert valid
 bs=max(valid,key=lambda x:x['samples_per_second'])['batch']
 write(R/'records'/f'{a.model}_scoring_audit.json',dict(upstream=audits,throughput=timings,selected_batch=bs,scope='Real unmodified HFLM loglikelihood and upstream process_results on same BF16 merged endpoint; intervention is custom and not official end-to-end benchmark run.'))
 def project(d,v):return (d@v)@v.T
 def equal_norm(d,reference):
  scale=float(reference.norm()/d.norm().clamp_min(1e-12));return d*min(scale,p['equal_norm_scale_cap']),scale
 with out.open('x') as f:
  for mode in (['cross_language'] if a.model=='Base' else ['cross_language','english_restore']):
   lang='sw' if mode=='cross_language' else 'en'
   for i,r in enumerate(rows):
    recipient=base_states[lang][i]
    source=base_states['en'][i] if mode=='cross_language' else source_base['en'][i]
    wrong_source=base_states['en'][r['unrelated_same_label']] if mode=='cross_language' else source_base['en'][r['unrelated_same_label']]
    delta=source-recipient;shared=project(delta,q)
    mean_delta=own['L18_boundary.mean_en']-own['L18_boundary.mean_sw'] if mode=='cross_language' else bases['L18_boundary.mean_en']-own['L18_boundary.mean_en']
    variants=[('none',torch.zeros_like(delta),0.,1.)]
    for alpha in [.5,1.]:
     for label,d in [('shared',shared),('centered_shared',project(delta-mean_delta,q)),('mean_only',project(mean_delta,q)),('unrelated_same_label',project(wrong_source-recipient,q)),('full',delta),('historical_mean_space',project(delta,old)),('random1115',project(delta,randoms[0]))]:variants.append((label,d*alpha,alpha,1.))
    for seed,v in zip(p['random_seeds'],randoms):
     rd=project(delta,v);eq,scale=equal_norm(rd,shared)
     if seed!=1115:variants.append((f'random{seed}',rd*.5,.5,1.))
     variants.append((f'random{seed}_equalnorm',eq*.5,.5,scale))
    comp,scale=equal_norm(delta-shared,shared);variants.append(('complement_equalnorm',comp*.5,.5,scale))
    all_lp=[]
    for j in range(0,len(variants),bs):
     lp,_=forward(*encoded[i,lang],torch.stack([v[1] for v in variants[j:j+bs]]));all_lp.extend(lp)
    null_error=float((all_lp[0]-baseline[i,lang]).abs().max())
    # A failed null control invalidates the whole row; never silently accept a changed baseline.
    if null_error>=.05 or int(all_lp[0].argmax())!=int(baseline[i,lang].argmax()):
     all_lp=[forward(*encoded[i,lang],v[1][None])[0][0] for v in variants]
     null_error=float((all_lp[0]-baseline[i,lang]).abs().max());assert null_error<.05 and int(all_lp[0].argmax())==int(baseline[i,lang].argmax())
     fallback=True
    else:fallback=False
    for (label,dose,alpha,scale),lp in zip(variants,all_lp):
     result=tasks[lang].process_results(r[lang],[(float(z),False) for z in lp]);gold=r['gold'];others=[j for j in range(4) if j!=gold]
     record=dict(model=a.model,mode=mode,id=r['id'],source_index=i,source_control_index=r['unrelated_same_label'],gold=gold,condition=label,alpha=alpha,pred=int(lp.argmax()),acc=float(result['acc']),margin=float(lp[gold]-lp[others].max()),logprobs=lp.tolist(),injection_norm=float(dose.norm()),unclipped_scale=scale,scale_clipped=scale>p['equal_norm_scale_cap'],null_max_logprob_error=null_error,single_batch_fallback=fallback,recipient_relative_drift=float(delta.square().sum()/recipient.square().sum()),shared_drift_fraction=float(shared.square().sum()/delta.square().sum().clamp_min(1e-12)))
     f.write(json.dumps(record)+'\n')
    f.flush()
    if i%32==0:print(json.dumps(dict(model=a.model,stage=mode,rows=i+1,seconds=time.monotonic()-start)),flush=True)
 write(R/'analysis'/f'intervention_{a.model}_DONE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated(),gpu=uuid,pid=os.getpid(),records=sum(1 for _ in out.open()),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),batch=bs))
if __name__=='__main__':main()
