"""Observed costs only. Separate incremental, reused and abandoned work."""
from pathlib import Path
import datetime,json
R=Path('/path/to/project');Q=R/'research/function_aware_formal_20261109';K=Path('/path/to/storage/checkpoints/function_aware_formal_20261109');L=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109')
def load(p):return json.loads(p.read_text())
def main():
 records=[]
 def add(kind,name,source,seconds=None,**extra):records.append(dict(kind=kind,name=name,source=str(source),observed_seconds=seconds,**extra))
 for path in sorted(K.glob('*/TRAINING_DONE.json')):
  d=load(path);acc=load(path.parent/'accounting-rank0.json');old=d.get('old',{})
  add('incremental_formal_training',path.parent.name,path,d['elapsed_seconds'],target_samples=d['exposure']['samples'],input_tokens=d['exposure']['input_tokens'],assistant_tokens=d['exposure']['shifted_assistant_tokens'],peak_allocated_bytes=d['peak_allocated_bytes'],trainable_parameters=acc['parameters']['trainable_unique_parameters'],old_build_samples=old.get('samples',0),old_forward_backward_calls=old.get('additional_forward_backward_calls',0),old_measured_seconds_including_sync=old.get('seconds_including_sync',0),timing_note='Old timing is nested inside total; never add it twice. Wall time is not CUDA kernel time.')
 for path in sorted(K.glob('*/SMOKE_DONE.json')):
  d=load(path);add('incremental_engineering',path.parent.name,path,d.get('elapsed_seconds'),target_samples=d['exposure']['samples'],benchmark_selection=False)
 cache=Path('/path/to/storage/datasets/function-aware-formal-20261109/features-v1/manifest.json')
 if cache.exists():add('incremental_common_teacher_cache','72_reference_and_normalizer',cache,load(cache)['elapsed_seconds'],shared_across_languages_and_methods=True)
 cache=Path('/path/to/storage/datasets/crosslayer-formal-20261109/teacher-v1/manifest.json')
 if cache.exists():add('reused_teacher_cache','dense_logit_KD_reference',cache,load(cache)['elapsed_seconds'],shared_across_languages=True,incremental_recompute_seconds=0)
 # Exact reused SW runs contain more material than the selected build224; report
 # their full historical cost, never prorate it down to make calibration cheaper.
 selected=['all_layer']+['anatomical_'+g+'_'+kind for g in ['early_attention','early_mlp','middle_attention','middle_mlp','late_attention','late_mlp'] for kind in ['group','uniform']]
 folder=Path('/path/to/logs/qwen3-multilingual/functional-risk-p3-20261109/ce-mc-v3');used=[]
 for name in selected:
  path=folder/name/'DONE.json'
  if path.exists():used.append(load(path))
 add('reused_sw_calibration','13_existing_CE_runs',folder,sum(x['elapsed_seconds'] for x in used) if len(used)==13 else None,completed_variants=len(used),source_rows=sum(x['rows'] for x in used),used_build_rows_per_variant=224,incremental_recompute_seconds=0,includes_original_unused_check_compute=True)
 for lang in ['ar','ru']:
  out=L/(lang+'_calibration');path=out/'worker.log';parts=[]
  if path.exists():
   for line in path.open():
    try:x=json.loads(line)
    except (ValueError,TypeError):continue
    if 'variant' in x and 'elapsed_seconds' in x:parts.append(x)
  add('incremental_calibration',lang,path,sum(x['elapsed_seconds'] for x in parts) if parts else None,completed_variants=len(parts),expected_variants=13,rows_per_variant=224,status='COMPLETE' if len(parts)==13 else 'PENDING_OR_RUNNING',timing_note='Observed variant forward-loop wall times; excludes loading/queue, not assumed GPU kernel time')
 for lang,info in load(Q/'reuse_audit.json').items():
  path=Path(info['original_adapter'])/'TRAINING_DONE.json';d=load(path)
  add('reused_calibration_adapter_training',lang,path,d.get('elapsed_seconds'),target_samples=d['exposure']['samples'],incremental_recompute_seconds=0,comparison_note='Historical calibrator training is part of resource provenance; aligned AR/RU ordinary controls are separately reported, not reused as identical microbatch runs')
 cache=Path('/path/to/storage/datasets/crosslayer-formal-20261109/sst-v1/DONE.json')
 if cache.exists():
  d=load(cache);add('abandoned_previous_plan','unused_SST_basis',cache,d.get('elapsed_seconds'),used_by_current_method=False,not_charged_as_candidate_training=True)
 result=dict(status='PARTIAL_OBSERVED_COSTS',updated_at=datetime.datetime.now().astimezone().isoformat(),records=records,not_recorded_is_not_zero=True,excludes_unfinished_training=True,caveats=['Queue waiting is separate from training wall time.','Reused historical costs are not zero compute; no newly rerun teacher/calibration is implied.','Shared caches are listed once, not multiplied by every method.','CPU offload, model loading and saving are included only according to their source timers.','Training metadata FLOPs estimates omit custom old-model work and must not be called total measured FLOPs.','Final compute comparison requires every method complete; no speed advantage claimed yet.'])
 (Q/'compute_costs.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(records=len(records),status=result['status'])))
if __name__=='__main__':main()
