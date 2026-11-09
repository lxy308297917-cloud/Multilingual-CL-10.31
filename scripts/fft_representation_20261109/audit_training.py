"""Independent CPU audit of the two new fixed-Final400 FFT endpoints."""
from pathlib import Path
import collections,json,math
import yaml
from audit_full_updates import sha
R=Path('/path/to/project');Q=R/'research/fft_representation_20261109';C=R/'configs/fft_representation_20261109';K=Path('/path/to/storage/checkpoints/fft_representation_20261109');H=Path('/path/to/storage/checkpoints/aya_4b_20261105/fft_final_v2');D=Path('/path/to/storage/datasets/functional-representation-20261109/p2-v1')
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.open()]
def main():
 reference=load(H/'TRAINING_DONE.json');source=rows(H/'exposure-rank0.jsonl');source_lrs=rows(H/'applied-lr-rank0.jsonl');old=rows(D/'build_pool.jsonl');manifest=load(D/'manifest.json')
 assert len(source)==6400 and len(source_lrs)==400 and len(old)==128
 configs=[yaml.safe_load((C/(mode+'_final.yaml')).read_text()) for mode in ['none','full36']]
 def scientific(cfg):return {k:v for k,v in cfg.items() if k not in ['output_dir','logging_dir']}
 assert scientific(configs[0])==scientific(configs[1]),'New protected/unprotected target recipes differ'
 models=[];pending=[]
 for mode,cfg in zip(['none','full36'],configs):
  folder=K/(mode+'_final_v1');marker=folder/'TRAINING_DONE.json'
  if not marker.exists():pending.append(mode);continue
  done=load(marker);assert done['status']=='TRAINING_DONE' and done['mode']==mode and done['steps']==cfg['max_steps']==400
  assert cfg['finetuning_type']=='full' and cfg['per_device_train_batch_size']==1 and cfg['gradient_accumulation_steps']==16 and cfg['learning_rate']==1e-5
  assert cfg['seed']==cfg['data_seed']==42 and cfg['cutoff_len']==2048 and not cfg['packing']
  assert cfg['bf16'] and cfg['gradient_checkpointing_kwargs']=={'use_reentrant':False} and cfg['use_reentrant_gc'] is False
  assert done['config_sha256']==sha(C/(mode+'_final.yaml')) and done['source_old_manifest_sha256']==sha(D/'manifest.json')
  assert done['normalizers']==manifest['normalizers'] and done['lambda_repr']==1 and done['old_check_samples']==0
  assert done['backward_calls']==6400 and done['applied_cpuadam_updates']==400 and done['non_reentrant_checkpointing'] and done['saved_optimizer_state']
  actual=rows(folder/'exposure-rank0.jsonl');assert len(actual)==6400 and [x['input_label_sha256'] for x in actual]==[x['input_label_sha256'] for x in source]
  assert done['exposure']==reference['exposure'],'Target token or padding exposure changed despite matched microbatch'
  acc=load(folder/'accounting-rank0.json');assert acc['step']==400 and acc['global_counts']==done['exposure']
  assert acc['parameters']['unique_model_parameters']==acc['parameters']['trainable_unique_parameters']==4022468096
  grad=load(folder/'gradient_coverage.json');assert grad['all_expected_parameters_received_finite_gradient'] and grad['parameter_tensors']==398
  assert len(grad['gradient_norms'])==398 and all(math.isfinite(v) for v in grad['gradient_norms'].values())
  assert 'model.embed_tokens.weight' in grad['gradient_norms'] and 'model.norm.weight' in grad['gradient_norms']
  bind=load(folder/'optimizer-binding.json');assert bind['actual_optimizer_type']=='DeepSpeedCPUAdam' and bind['shared_group_dictionaries_after']
  lrs=rows(folder/'applied-lr-rank0.jsonl');assert len(lrs)==400
  assert [x['actual_lrs'] for x in lrs]==[x['actual_lrs'] for x in source_lrs]
  assert all(x['actual_lrs']==x['scheduler_last_lrs'] for x in lrs)
  backward=load(folder/'joint_backward_audit.json');assert backward['backend_backward_calls']==backward['target_microbatches']==6400 and backward['gradient_accumulation']==16
  trace=rows(folder/'old_exposure.jsonl');assert len(trace)==done['old_samples']==(3200 if mode=='full36' else 0)
  if trace:
   assert [x['id'] for x in trace]==[old[i%128]['id'] for i in range(3200)]
   assert dict(collections.Counter(x['capability'] for x in trace))=={'math':1600,'instruction':1600}
   assert all(x['split']=='build' and x['layers']==36 and x['step']==1+i//8 and x['micro']==2*(i%8) for i,x in enumerate(trace))
   assert all(x['old_coefficient_inside_compute_loss']==(.125 if x['target_global_token_normalization'] else 2.) and math.isfinite(x['old_objective']) for x in trace)
  checkpoints=[]
  for step in [100,200,300,400]:
   path=folder/('checkpoint-'+str(step));assert load(path/'trainer_state.json')['global_step']==step
   assert (path/'model.safetensors.index.json').exists()
   optimizer=[x for x in path.rglob('*optim_states.pt') if x.is_file() and x.stat().st_size>0];assert optimizer,'Declared optimizer checkpoint is missing'
   checkpoints.append(dict(step=step,optimizer_files=len(optimizer),optimizer_bytes=sum(x.stat().st_size for x in optimizer)))
  assert done['weights_sha256'] and all(sha(folder/name)==digest for name,digest in done['weights_sha256'].items())
  assert load(folder/'config.json')['tie_word_embeddings'] is True and not (folder/'adapter_config.json').exists()
  models.append(dict(mode=mode,verified_final400=True,target_exact_historical_input_label_stream=True,target_exposure=done['exposure'],unique_trainable_parameters=4022468096,first_step_all_398_parameter_tensors_received_finite_gradient=True,one_backward_per_target_microbatch=True,applied_learning_rate_trace_matches_historical_400=True,old_build_samples=len(trace),old_check_samples=0,old_reference_prefix_tokens=sum(x['reference_prefix_tokens'] for x in trace),old_feature_prediction_positions=sum(x['protected_positions'] for x in trace),old_reference_prefix_not_CE_supervision=True,checkpoints=checkpoints,optimizer_state_saved_not_a_resume_execution_test=True,weights_sha256=done['weights_sha256'],config_sha256=done['config_sha256'],elapsed_seconds=done['elapsed_seconds'],peak_allocated_bytes=done['peak_allocated_bytes']))
 report=dict(status='BOTH_NEW_FFT_ENDPOINTS_INDEPENDENTLY_AUDITED' if len(models)==2 else 'INCOMPLETE',verified_count=len(models),expected=2,pending=pending,models=models,new_protected_unprotected_scientific_recipes_identical=True,historical_implementation_difference='Both new runs use non-reentrant checkpointing and the same combined-backward wrapper; historical FFT remains separately identified.',additional_gpu_jobs=0)
 (Q/'training_artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],verified=len(models),pending=pending)))
if __name__=='__main__':main()
