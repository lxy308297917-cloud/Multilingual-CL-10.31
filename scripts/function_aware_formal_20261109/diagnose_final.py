"""Final-only auxiliary diagnostics, gated behind every queued formal benchmark.

No training, coefficient fitting or checkpoint choice. P1 check128 was exposed
in older research, but is disjoint from the formal training build128.
"""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,math,os,sys,time
R=Path('/path/to/project')
P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
M=Path('/path/to/models/Qwen3-4B')
D=Path('/path/to/storage/datasets/functional-representation-20261109/p1-v1')
O=Path('/path/to/storage/datasets/formal-representation-diagnostics-20261109/v1')
L=Path('/path/to/logs/qwen3-multilingual/formal-representation-diagnostics-20261109')
C=R/'configs/function_aware_formal_20261109/diagnostic_models.json'
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.open()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def barrier():
 fa=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109/evaluation');fft=Path('/path/to/logs/qwen3-multilingual/fft-representation-20261109/evaluation')
 jobs=load(R/'configs/function_aware_formal_20261109/jobs.json')
 needed=[fa/name/stage/'DONE.json' for name in jobs for stage in ['reading','summary','gmmlu','mmlu','core','code']]
 needed += [fa/'sw_all/mmlu/DONE.json',fa/'ar_Base/gmmlu/DONE.json',fa/'ru_Base/gmmlu/DONE.json']
 needed += [fft/p/stage/'DONE.json' for p in ['matched_fft','fft_full36'] for stage in ['reading','summary','gmmlu','mmlu','core','code','mt_en_sw','mt_sw_en']]
 needed += [fft/p/'code/DONE.json' for p in ['historical_fft','historical_ssu']]
 needed += [fft/p/'mmlu/DONE.json' for p in ['p2_full36','p2_ce']]
 needed += [fft/p/stage/'DONE.json' for p in ['p2_all','p2_full36','p2_ce'] for stage in ['mt_en_sw','mt_sw_en']]
 missing=[p for p in needed if not p.exists()]
 if not missing:assert all(load(p)['status']=='FIXED_ENDPOINT_STAGE_COMPLETE_PENDING_PAIRED_AUDIT' for p in needed)
 return len(missing)
def prepare():
 assert not C.exists() and not O.exists() and not L.exists();O.mkdir(parents=True);L.mkdir(parents=True)
 old=load(Path('/path/to/storage/datasets/functional-representation-20261109/p2-v1/manifest.json'))
 assert sha(D/'encoded_selected.jsonl')==old['source_encoded_sha256']
 selected=[x for x in rows(D/'encoded_selected.jsonl') if x['split']=='check'];assert len(selected)==128
 assert dict(collections.Counter(x['capability'] for x in selected))=={'math':64,'instruction':64}
 assert not {x['id'] for x in selected}.intersection(old['build_ids'])
 (O/'check128.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected))
 models={'Base':dict(kind='base',artifact=str(M),report_family='function_aware_formal_20261109')}
 for lang in ['sw','ar','ru']:
  for method in ['all','full','kd','random','function']:
   artifact=Path('/path/to/storage/checkpoints/functional_representation_20261109/p2/all_layer_final_v1') if lang=='sw' and method=='all' else Path('/path/to/storage/checkpoints/function_aware_formal_20261109')/(lang+'_'+method+'_final_v1')
   models[lang+'_'+method]=dict(kind='lora',artifact=str(artifact),report_family='function_aware_formal_20261109')
 for name,profile in load(R/'configs/fft_representation_20261109/evaluation_profiles.json').items():
  if name=='p2_all':continue # Same artifact already appears as sw_all.
  models[name]=dict(kind='lora' if profile['kind']=='lora' else 'full',artifact=profile['artifact'],report_family='fft_representation_20261109')
 C.write_text(json.dumps(models,indent=2)+'\n')
 (O/'manifest.json').write_text(json.dumps(dict(status='CPU_PREPARED_GPU_NOT_RUN',rows=128,class_counts={'math':64,'instruction':64},data_sha256=sha(O/'check128.jsonl'),p1_encoded_sha256=sha(D/'encoded_selected.jsonl'),p1_reference_sha256=old['source_reference_sha256'],models_sha256=sha(C),diagnostic_source_sha256=sha(Path(__file__)),purpose='Auxiliary final-only drift and reference-answer NLL; not downstream accuracy, not selection',data_use='Check128 excluded from formal training; historically exposed P1 diagnostic material',gpu_barrier='All main14 formal six-stage evaluations plus SW all MMLU, AR/RU Base GM and all26 FFT stages complete',no_hyperparameter_selection=True),indent=2)+'\n')
 print(json.dumps(dict(status='PREPARED_NOT_RUN',models=len(models),missing_formal_stages=barrier())))
def worker(model_id,uuid):
 assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['AYA_EXPECTED_GPU_UUID']==uuid
 assert barrier()==0,'Formal benchmarks have absolute priority over this auxiliary job'
 import torch
 from transformers import AutoModelForCausalLM
 from peft import PeftModel
 from safetensors.torch import load_file,save_file
 torch.set_num_threads(4)
 assert str(torch.cuda.get_device_properties(0).uuid)==uuid.removeprefix('GPU-')
 manifest=load(O/'manifest.json');assert sha(O/'check128.jsonl')==manifest['data_sha256'] and sha(C)==manifest['models_sha256'] and sha(Path(__file__))==manifest['diagnostic_source_sha256']
 definitions=load(C);spec=definitions[model_id];artifact=Path(spec['artifact']);identity={}
 if spec['kind']!='base':
  done=load(artifact/'TRAINING_DONE.json');assert done['steps']==400 and done['exposure']['samples']==6400;identity=done['weights_sha256'];assert all(sha(artifact/name)==value for name,value in identity.items())
 else:
  lock=load(R/'configs/aya_4b_20261105/model.lock.json');assert all(sha(M/name)==v['sha256'] for name,v in lock['files'].items());identity={name:v['sha256'] for name,v in lock['files'].items()}
 model=AutoModelForCausalLM.from_pretrained(M if spec['kind']=='lora' else artifact,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
 if spec['kind']=='lora':model=PeftModel.from_pretrained(model,artifact,is_trainable=False)
 model.eval().requires_grad_(False);raw=model.get_base_model() if hasattr(model,'get_base_model') else model
 assert len(raw.model.layers)==36
 selected=rows(O/'check128.jsonl');cache={} if model_id=='Base' else load_file(str(O/'check128_reference72.safetensors'))
 normal36=load(Path('/path/to/storage/datasets/functional-representation-20261109/p2-v1/manifest.json'))['normalizers'];normal72=load(Path('/path/to/storage/datasets/function-aware-formal-20261109/features-v1/manifest.json'))['normalizers']
 previous=None;index=None
 if model_id=='Base':
  assert sha(D/'base_features_v1/base_features.safetensors')==manifest['p1_reference_sha256'];previous=load_file(str(D/'base_features_v1/base_features.safetensors'))['features'];index={x['id']:x for x in load(D/'base_features_v1/rows.json')}
 out=L/model_id;out.mkdir(exist_ok=False);records=[];oracle=0.;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 with torch.inference_mode():
  for row in selected:
   ids=torch.tensor([row['input_ids']],device='cuda');positions=[row['positions'][slot] for slot in row['answer_prediction_slots']];states={};hooks=[]
   for layer,block in enumerate(raw.model.layers):
    def pre(module,inputs,layer=layer):states[2*layer]=inputs[0][0,positions,:]
    def post(module,inputs,output,layer=layer):states[2*layer+1]=(output[0] if isinstance(output,tuple) else output)[0,positions,:]
    hooks.append(block.post_attention_layernorm.register_forward_pre_hook(pre));hooks.append(block.register_forward_hook(post))
   try:hidden=raw.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
   finally:
    for hook in hooks:hook.remove()
   current=torch.stack([states[i] for i in range(72)],dim=1).float()
   if model_id=='Base':
    cache[row['id']]=current.cpu().to(torch.bfloat16).contiguous();offset=index[row['id']]['feature_offset_start'];old=previous[[offset+slot for slot in row['answer_prediction_slots']]].to('cuda').float();error=float((current[:,1::2]-old).abs().max());oracle=max(oracle,error);assert error<1e-3
   reference=cache[row['id']].to('cuda').float();drift=(current-reference).square().sum((0,2))/reference.square().sum((0,2)).clamp_min(1e-12)
   prefix=row['prefix_length'];logits=raw.lm_head(hidden[:,prefix-1:-1,:]).float();nllsum=float(torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,prefix:].reshape(-1),reduction='sum'));count=ids.shape[1]-prefix;assert count>0 and math.isfinite(nllsum) and torch.isfinite(drift).all()
   records.append(dict(id_sha256=hashlib.sha256(row['id'].encode()).hexdigest(),capability=row['capability'],assistant_tokens=count,assistant_nll=nllsum/count,per_site_relative_squared_drift=drift.cpu().tolist()))
   del ids,hidden,logits,current,reference,drift,states
  torch.cuda.synchronize()
 elapsed=time.monotonic()-started
 if model_id=='Base':save_file(cache,str(O/'check128_reference72.safetensors'))
 summary={}
 for cap in ['math','instruction']:
  group=[x for x in records if x['capability']==cap];means=[sum(x['per_site_relative_squared_drift'][i] for x in group)/64 for i in range(72)]
  summary[cap]=dict(examples=64,mean_per_example_assistant_nll=sum(x['assistant_nll'] for x in group)/64,token_weighted_assistant_nll=sum(x['assistant_nll']*x['assistant_tokens'] for x in group)/sum(x['assistant_tokens'] for x in group),per_site72_relative_squared_drift=means,per_layer36_relative_squared_drift=means[1::2],six_group_drift=[sum(means[i] for i in range(72) if 2*(i//24)+i%2==g)/12 for g in range(6)],normalized_loss72=sum(means)/72/normal72[cap],normalized_loss36=sum(means[1::2])/36/normal36[cap])
 if spec['kind']!='base':assert all(sha(artifact/name)==value for name,value in identity.items())
 (out/'per_example.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
 result=dict(status='FINAL_ONLY_AUXILIARY_DIAGNOSTICS_COMPLETE',model_id=model_id,kind=spec['kind'],endpoint='Base' if model_id=='Base' else 'Final400',check128_used_for_training=0,historical_check_exposure=True,auxiliary_only=True,no_downstream_benchmark_claim=True,model_identity=identity,data_manifest_sha256=sha(O/'manifest.json'),teacher_reference_sha256=sha(O/'check128_reference72.safetensors'),teacher36_oracle_max_error=oracle if model_id=='Base' else None,summary=summary,elapsed_seconds=elapsed,timing_scope='Synchronized check128 forward-loop metrics; excludes loading and cache serialization',peak_allocated_bytes=torch.cuda.max_memory_allocated(),per_example_sha256=sha(out/'per_example.jsonl'))
 (out/'DONE.json').write_text(json.dumps(result,indent=2)+'\n');target=R/'research'/spec['report_family']/'diagnostics';target.mkdir(exist_ok=True);(target/(model_id+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(status=result['status'],model_id=model_id,elapsed_seconds=elapsed)),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',action='store_true');ap.add_argument('--model-id');ap.add_argument('--gpu-uuid');a=ap.parse_args()
 if a.prepare:prepare();return
 if a.worker:worker(a.model_id,a.gpu_uuid);return
 definitions=load(C);deadline=time.monotonic()+120*3600
 while barrier():
  (L/'schedule.json').write_text(json.dumps(dict(status='WAITING_ALL_FORMAL_BENCHMARKS_NO_GPU_RESERVED',missing_formal_stages=barrier(),updated_at=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
  if time.monotonic()>deadline:raise TimeoutError('Finite post-benchmark barrier expired; no GPU acquired')
  time.sleep(60)
 sys.path.insert(0,str(R/'scripts/aya_4b_multilang_20261106'));from gpu_slot import run_job
 env=dict(os.environ,HF_HOME='/path/to/storage/hf-cache-qwen3',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4')
 completed=[]
 for name in definitions:
  def save(status,**more):(L/'schedule.json').write_text(json.dumps(dict(status=status,model_id=name,completed=completed,updated_at=datetime.datetime.now().astimezone().isoformat(),**more),indent=2)+'\n')
  try:run_job(lambda uuid:[P,str(Path(__file__).resolve()),'--worker','--model-id',name,'--gpu-uuid',uuid],env,L/(name+'.log'),save)
  except Exception as exc:save('FAILED',error=repr(exc));raise
  assert load(L/name/'DONE.json')['status']=='FINAL_ONLY_AUXILIARY_DIAGNOSTICS_COMPLETE';completed.append(name)
 (L/'schedule.json').write_text(json.dumps(dict(status='ALL_AUXILIARY_DIAGNOSTICS_COMPLETE_PENDING_REPORT',completed=completed),indent=2)+'\n')
if __name__=='__main__':main()
