"""CPU audit of complete fixed endpoints; no inference, retries or model selection."""
from pathlib import Path
import collections,hashlib,json,re,time,datetime
import torch,yaml
from safetensors.torch import load_file
R=Path('/path/to/project');C=R/'configs/function_aware_formal_20261109';Q=R/'research/function_aware_formal_20261109';K=Path('/path/to/storage/checkpoints/function_aware_formal_20261109')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).open()]
def main():
 torch.set_num_threads(4);started=time.monotonic();jobs=json.loads((C/'jobs.json').read_text());reports=[];pending=[];initial={};norm_rows=[]
 reference=Path('/path/to/storage/checkpoints/functional_representation_20261109/p2/all_layer_final_v1/checkpoint-0/adapter_model.safetensors')
 assert reference.exists();initial['sw_all_reused']=sha(reference)
 for name,job in jobs.items():
  folder=K/(name+'_final_v1');init=folder/'checkpoint-0/adapter_model.safetensors'
  if init.exists():initial[name]=sha(init)
  if not (folder/'TRAINING_DONE.json').exists():pending.append(name);continue
  final=json.loads((folder/'TRAINING_DONE.json').read_text());cfg=yaml.safe_load((C/(name+'_final.yaml')).read_text());adapter=json.loads((folder/'adapter_config.json').read_text())
  assert cfg['max_steps']==final['steps']==400 and cfg['per_device_train_batch_size']==2 and cfg['gradient_accumulation_steps']==8
  assert cfg['seed']==cfg['data_seed']==42 and cfg['learning_rate']==1e-4 and not cfg['packing'] and cfg['cutoff_len']==2048
  assert adapter['r']==16 and adapter['lora_alpha']==32 and adapter['lora_dropout']==.05 and adapter['bias']=='none'
  assert not adapter.get('use_dora') and not adapter.get('use_rslora') and not adapter.get('modules_to_save')
  assert final['config_sha256']==sha(C/(name+'_final.yaml'))
  assert final['mode']==job['method'] and final['language']==job['language']
  assert final['exposure']['samples']==6400 and final['target_order_matches_original_all6400']
  target=rows(folder/'exposure-rank0.jsonl');source=rows(job['source_exposure']);assert len(target)==len(source)==6400
  assert [x['input_label_sha256'] for x in source]==[x['input_label_sha256'] for x in target]
  assert all((folder/f'checkpoint-{s}/adapter_model.safetensors').exists() for s in [0,50,100,200,300,400])
  final_state=json.loads((folder/'checkpoint-400/trainer_state.json').read_text());assert final_state['global_step']==400
  expected=final['weights_sha256']['adapter_model.safetensors'];assert sha(folder/'adapter_model.safetensors')==expected==sha(folder/'checkpoint-400/adapter_model.safetensors')
  trace=rows(folder/'old_exposure.jsonl');count=0 if job['method']=='all' else 3200;assert len(trace)==count
  oldrows=rows('/path/to/storage/datasets/functional-representation-20261109/p2-v1/build_pool.jsonl')
  if count:
   assert [x['id'] for x in trace]==[oldrows[i%128]['id'] for i in range(count)]
   assert all(x['split']=='build' and x['backward_loss_divisor']==8 and x['coefficient']==1 for x in trace)
   assert dict(collections.Counter(x['capability'] for x in trace))=={'math':1600,'instruction':1600}
  weight_provenance=None
  if job['method'] in ['random','function']:
   path=C/(job['language']+'_weights.json');frozen=json.loads(path.read_text());calibration=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109')/(job['language']+'_calibration')
   assert sha(path)==sha(calibration/'DONE.json')
   assert frozen['status']=='WEIGHTS_FROZEN_NO_PERFORMANCE_GATE' and frozen['language']==job['language']
   assert frozen['calibration_raw_sha256']==sha(calibration/'predictions.jsonl')
   values=frozen[job['method']];assert len(values)==6 and abs(sum(values)-6)<1e-8 and min(values)>=.5-1e-8 and max(values)<=2+1e-8
   assert frozen['random']==[frozen['function'][i] for i in frozen['random_permutation']]
   assert path.stat().st_mtime<=datetime.datetime.fromisoformat(final['started_at']).timestamp()
   weight_provenance=dict(frozen_input_sha256=sha(path),selected_six_weights=values,calibration_raw_sha256=frozen['calibration_raw_sha256'],weight_file_predates_training=True,source='Independent post-run audit of immutable input used by the frozen loss code; not a newly invented in-training log')
  weights=load_file(str(folder/'adapter_model.safetensors'));keys=[k for k in weights if k.endswith('lora_A.weight')];assert len(keys)==252 and len(weights)==504
  total=0.
  for key in sorted(keys):
   a=weights[key].double();b=weights[key.replace('lora_A','lora_B')].double();value=float(((b.T@b)*(a@a.T).T).sum()*4);assert value>=0
   total+=value;norm_rows.append(dict(name=name,module=key.removesuffix('.lora_A.weight'),effective_update_squared_frobenius=value))
  reports.append(dict(name=name,method=job['method'],language=job['language'],verified_final400=True,adapter_sha256=expected,initial_adapter_sha256=initial[name],exact_target_input_label_stream_6400=True,old_build_samples=count,old_check_samples=0,old_input_tokens=sum(x['input_tokens'] for x in trace),old_reference_prefix_tokens=sum(x['assistant_tokens'] for x in trace),effective_delta_frobenius=total**.5,method_weight_input_provenance=weight_provenance,elapsed_seconds=final['elapsed_seconds'],peak_allocated_bytes=final['peak_allocated_bytes'],historical_configuration_matches_source=final['original_exposure_sha256']==sha(job['source_exposure'])))
 result=dict(status='ALL_NEW_ENDPOINTS_AUDITED' if len(reports)==14 else 'INCOMPLETE_RUNNING',verified_count=len(reports),expected=14,pending=pending,initial_weight_sha256=initial,observed_initial_weights_identical=len(set(initial.values()))==1,all_initials_observed=len(initial)==15,models=reports,audit_cpu_seconds=time.monotonic()-started,extra_gpu_jobs=0,no_effectiveness_claim_from_norms=True)
 (Q/'training_artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 if reports:
  import csv
  with (Q/'effective_update_norms.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(norm_rows[0]));w.writeheader();w.writerows(norm_rows)
 print(json.dumps(dict(status=result['status'],verified=len(reports),pending=len(pending),observed_initial_weights_identical=result['observed_initial_weights_identical'],observed_initials=len(initial))))
if __name__=='__main__':main()
