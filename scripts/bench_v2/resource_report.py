"""Summarize measured process wall times and sampled memory, not kernel GPU time."""
import argparse,datetime,json,os
from pathlib import Path
from common import dump

def timestamp(x):return datetime.datetime.fromisoformat(x).timestamp()
def main():
 p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 obs=[json.loads(x) for x in a.observations.read_text().splitlines()];runs=[];uid=os.getuid()
 for rec in json.loads(a.jobs.read_text()):
  run=Path(rec['output']);directory=Path(rec['launch']).parent;startfile=directory/'process.json';exitfile=directory/'exit.json'
  if not (startfile.exists() and exitfile.exists()):continue
  start=json.loads(startfile.read_text());end=json.loads(exitfile.read_text());lo=timestamp(start['started_at']);hi=timestamp(end['finished_at']);measure=[]
  for item in obs:
   if not lo<=timestamp(item['time'])<=hi:continue
   same=[x for x in item.get('processes',[]) if x['gpu_uuid']==rec['gpu_uuid'] and x['uid']==uid]
   memory=sum(int(x['used_memory_mib']) for x in same if str(x['used_memory_mib']).isdigit())
   if same:measure.append(memory)
  cost=json.loads((run/'cost.json').read_text()) if (run/'cost.json').exists() else {};scorepaths=list(run.glob('score-v*/summary.json'));score=json.loads(scorepaths[-1].read_text()) if scorepaths else {}
  runs.append(dict(run_id=run.name,method=rec['method'],benchmark=rec.get('task'),scope=rec['stage'],exit_code=end['exit_code'],process_wall_seconds=end['elapsed_seconds'],engine_wall_seconds=cost.get('elapsed_seconds'),generated_tokens=cost.get('generated_tokens'),input_tokens=cost.get('input_tokens'),sampled_own_process_memory_max_mib=max(measure) if measure else None,memory_samples=len(measure),scoring_cpu_process_wall_seconds=score.get('scoring_cpu_wall_seconds'),started_at=start['started_at'],finished_at=end['finished_at']))
 a.output.mkdir(parents=True);dump(a.output/'resource_report.json',dict(runs=runs,total_process_wall_seconds=sum(x['process_wall_seconds'] for x in runs),total_engine_wall_seconds=sum(x['engine_wall_seconds'] or 0 for x in runs),total_measured_scoring_process_wall_seconds=sum(x['scoring_cpu_process_wall_seconds'] or 0 for x in runs),peak_sampled_own_process_memory_mib=max((x['sampled_own_process_memory_max_mib'] or 0 for x in runs),default=0) or None,failed_attempts=sum(x['exit_code']!=0 for x in runs),limitations=['Summed process/engine wall time is not CUDA kernel time, exclusive CPU time, or parallel batch elapsed time.','Memory monitoring starts partway through this study; missing is null, not zero. Sampled maxima are lower bounds on actual continuous peaks.','Own-process attribution uses Linux UID and assigned GPU within run wall interval; no other-user processes included.']))
 print(json.dumps({'completed_attempts':len(runs),'output':str(a.output)}))
if __name__=='__main__':main()
