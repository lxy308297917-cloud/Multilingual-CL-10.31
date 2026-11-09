"""Finite read-only progress supervision. No restarts, retries or model selection."""
from pathlib import Path
import datetime,json,os,subprocess,time
R=Path('/path/to/project');L=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109');Q=R/'research/function_aware_formal_20261109';P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
def load(p):return json.loads(p.read_text())
def main():
 log=L/'observer.log';deadline=time.monotonic()+120*3600
 while time.monotonic()<deadline:
  started=datetime.datetime.now().astimezone().isoformat();own=[]
  proc=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
  if proc.returncode==0:
   for line in proc.stdout.strip().splitlines():
    pid,uuid,memory=[x.strip() for x in line.split(',')]
    path=Path('/proc')/pid
    try:
     if path.stat().st_uid!=os.getuid():continue
     command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
     own.append(dict(pid=int(pid),gpu_uuid=uuid,used_mib=memory,command=command))
    except (FileNotFoundError,PermissionError):continue
  for script in ['progress.py','collect_results.py']:
   call=subprocess.run([P,str(Path(__file__).with_name(script))],cwd=R,capture_output=True,text=True)
   with log.open('a') as f:f.write(json.dumps(dict(at=started,script=script,returncode=call.returncode,stdout=call.stdout,stderr=call.stderr))+'\n')
   if call.returncode:raise RuntimeError('Read-only report failed; inspect '+script)
  progress=load(Q/'progress.json');schedule=load(L/'evaluation-schedule.json')
  failures=[]
  for p in L.glob('*/state.json'):
   state=load(p)
   if state.get('status')=='FAILED':failures.append(dict(job=p.parent.name,error=state.get('error')))
  reused=[]
  for item in schedule.get('failed',[]):
   marker=L/'evaluation'/item['name']/item['stage']/'REUSED_RESULT.json'
   if item['name']=='sw_Base' and item['stage']=='mmlu' and marker.exists() and load(marker).get('is_scientific_failure') is False:reused.append(item)
   else:failures.append(item)
  snapshot=dict(at=started,observer_pid=os.getpid(),own_gpu_processes=own,actual_account_gpu_count=len({x['gpu_uuid'] for x in own}),gpu_query_returncode=proc.returncode,completed_training=progress['completed'],total_training=progress['new_runs'],evaluation_completed=len(schedule.get('completed',[])),evaluation_reused_instead_of_duplicate=len(reused),evaluation_active=len(schedule.get('running',[])),evaluation_pending=len(schedule.get('pending',[])),failures=failures,mutates_jobs=False)
  (L/'live_observation.json').write_text(json.dumps(snapshot,indent=2)+'\n')
  if progress['completed']==14 and not schedule.get('pending') and not schedule.get('running'):
   (Q/'READY_FOR_FINAL_AUDIT.json').write_text(json.dumps(dict(status='ALL_JOBS_TERMINAL_NOT_PROOF_OF_SUCCESS',observation=snapshot,still_required=['failure audit','raw per-item metric verification','paired intervals','update norms and representation drift','cost accounting','final report and curated GitHub publication']),indent=2)+'\n');return
  time.sleep(300)
 raise TimeoutError('Finite120h observer expired; existing tasks untouched')
if __name__=='__main__':main()
