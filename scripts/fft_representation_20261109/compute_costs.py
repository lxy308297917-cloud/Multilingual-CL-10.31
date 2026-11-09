"""Separate actual FFT training, engineering and execution costs; no estimates-as-data."""
from pathlib import Path
import datetime,json
R=Path('/path/to/project');Q=R/'research/fft_representation_20261109';K=Path('/path/to/storage/checkpoints/fft_representation_20261109');L=Path('/path/to/logs/qwen3-multilingual/fft-representation-20261109')
def load(p):return json.loads(p.read_text())
def main():
 profiles=load(R/'configs/fft_representation_20261109/evaluation_profiles.json');records=[]
 for name,profile in profiles.items():
  path=Path(profile['artifact'])/'TRAINING_DONE.json'
  if not path.exists():continue
  d=load(path);accpath=path.parent/'accounting-rank0.json';acc=load(accpath) if accpath.exists() else {}
  records.append(dict(profile=name,kind='incremental_formal' if name in ['matched_fft','fft_full36'] else 'reused_historical_training',source=str(path),elapsed_seconds=d.get('elapsed_seconds',acc.get('elapsed_this_process_seconds')),target_samples=d['exposure']['samples'],input_tokens=d['exposure']['input_tokens'],assistant_tokens=d['exposure']['shifted_assistant_tokens'],old_build_samples=d.get('old_samples',d.get('old',{}).get('samples',0)),peak_allocated_bytes=d.get('peak_allocated_bytes'),parameters=acc.get('parameters'),saved_optimizer_state=d.get('saved_optimizer_state'),incremental_recompute=name in ['matched_fft','fft_full36']))
 for path in sorted(K.glob('*_smoke_*/SMOKE_DONE.json')):
  d=load(path);records.append(dict(profile=path.parent.name,kind='incremental_engineering',elapsed_seconds=d['elapsed_seconds'],target_samples=d['exposure']['samples'],old_build_samples=d['old_samples'],peak_allocated_bytes=d['peak_allocated_bytes'],source=str(path),used_as_benchmark_result=False))
 for path in sorted((L/'evaluation').glob('*/*/execution/scores.json')):
  d=load(path)
  if d.get('status')=='COMPLETE':records.append(dict(profile=path.parents[2].name,kind='incremental_cpu_code_execution',source=str(path),elapsed_seconds=d['seconds_model_execution'],samples=d['n'],timing_scope='isolated model-program execution; excludes generation/loading/queue'))
 result=dict(status='PARTIAL_OBSERVED_COSTS',updated_at=datetime.datetime.now().astimezone().isoformat(),records=records,not_recorded_is_not_zero=True,unfinished_runs_excluded=True,reference36_cache_recomputed=False,reference36_cache_cost='Historical source cost retained separately; not recorded by this track, never invent zero.',timing_caveats=['Queue waiting is not training duration.','FFT optimizer save/offload and CPU initialization are included according to original timers.','Do not add nested old-loss timings to total twice.','Engineering uses longest examples and cannot reliably extrapolate complete training duration.','Reused LoRA/FFT/SSU timings came from different implementations and resource states.','Evaluation GPU durations not recorded in source are unknown, not zero.'])
 (Q/'compute_costs.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],records=len(records))))
if __name__=='__main__':main()
