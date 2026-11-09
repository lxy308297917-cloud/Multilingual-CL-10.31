"""Verify existing engineering artifacts only; never starts or selects a model."""
from pathlib import Path
import hashlib,json,datetime
R=Path('/path/to/project');Q=R/'research/fft_representation_20261109';K=Path('/path/to/storage/checkpoints/fft_representation_20261109')
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.open()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 models=[];pending=[]
 for mode in ['none','full36']:
  root=K/(mode+'_smoke_v1')
  if not (root/'SMOKE_DONE.json').exists():pending.append(mode);continue
  done=load(root/'SMOKE_DONE.json');grad=load(root/'gradient_coverage.json');binding=load(root/'optimizer-binding.json');lr=rows(root/'applied-lr-rank0.jsonl');old=rows(root/'old_exposure.jsonl');back=load(root/'joint_backward_audit.json')
  assert done['status']=='ENGINEERING_SMOKE_DONE' and done['mode']==mode and done['steps']==3 and done['exposure']['samples']==48
  assert done['backward_calls']==back['backend_backward_calls']==48 and done['applied_cpuadam_updates']==len(lr)==3
  assert grad['all_expected_parameters_received_finite_gradient'] and grad['parameter_tensors']==398
  assert binding['actual_optimizer_type']=='DeepSpeedCPUAdam' and binding['shared_group_dictionaries_after']
  assert all(x['actual_lrs']==x['scheduler_last_lrs'] for x in lr) and lr[1]['actual_lrs'][0]>0
  assert len(old)==done['old_samples']==(24 if mode=='full36' else 0) and done['old_check_samples']==0
  assert done['non_reentrant_checkpointing'] and done['peak_allocated_bytes']<24*1024**3
  if old:assert any(x['old_objective']>1e-10 for x in old) and all(x['split']=='build' for x in old)
  models.append(dict(mode=mode,verified_engineering_only=True,steps=3,target_samples=48,old_build_samples=len(old),old_check_samples=0,all_398_unique_parameter_tensors_received_finite_gradient=True,backward_calls=48,applied_learning_rates=[x['actual_lrs'] for x in lr],actual_weight_decays=binding['actual_weight_decays'],cpu_adam_scheduler_binding_verified=True,peak_allocated_bytes=done['peak_allocated_bytes'],elapsed_seconds=done['elapsed_seconds'],normalized_old_drift_range=[min(x['old_objective'] for x in old),max(x['old_objective'] for x in old)] if old else None,proof_sha256={f:sha(root/f) for f in ['SMOKE_DONE.json','gradient_coverage.json','joint_backward_audit.json','old_exposure.jsonl','applied-lr-rank0.jsonl','optimizer-binding.json']}))
 result=dict(status='BOTH_FFT_ENGINEERING_PASSED' if len(models)==2 else 'PARTIAL_ENGINEERING',updated_at=datetime.datetime.now().astimezone().isoformat(),models=models,pending=pending,scope='Engineering only; full400 starts fresh. No benchmark performance gate.',numerical_note='Preserve the protected initial nonzero feature-loss floor; do not call all nonzero loss adaptation-induced drift.',benchmark_results_used=False)
 (Q/'engineering_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],verified=len(models),pending=pending)))
if __name__=='__main__':main()
