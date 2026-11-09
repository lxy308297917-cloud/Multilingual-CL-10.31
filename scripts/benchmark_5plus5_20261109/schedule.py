"""Finite manually started supplement; no new training, one extra GPU worker."""
from pathlib import Path
import datetime,json,os,subprocess,time
R=Path('/path/to/project');C=R/'configs/benchmark_5plus5_20261109';L=Path('/path/to/logs/qwen3-multilingual/benchmark-5plus5-20261109');P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
def main():
 models=json.loads((C/'models.json').read_text());todo=[(x,'sib200') for x in models]+[('ru_historical_lora','gmmlu')];deadline=time.monotonic()+120*3600
 # First launched historical AR GM is a separate manually started job. Wait for
 # its actual terminal marker; do not run a second supplementary GPU worker.
 def save(status,**kw):(L/'schedule.json').write_text(json.dumps(dict(status=status,at=datetime.datetime.now().astimezone().isoformat(),pid=os.getpid(),pending=todo,**kw),indent=2)+'\n')
 first=L/'ar_historical_lora/gmmlu'
 while not (first/'DONE.json').exists():
  if (first/'state.json').exists() and json.loads((first/'state.json').read_text())['status']=='FAILED':save('FAILED_INITIAL_GM');return
  if time.monotonic()>deadline:raise TimeoutError('Finite supplement wait expired')
  save('WAITING_ALREADY_RUNNING_AR_GM');time.sleep(30)
 failures=[]
 while todo:
  progress=False
  for name,task in list(todo):
   out=L/name/task
   if (out/'DONE.json').exists():todo.remove((name,task));progress=True;continue
   if out.exists():
    # Preserve any terminal failure. Authorized recovery is diagnosed manually
    # into a distinct directory, never overwritten or blindly retried here.
    failures.append(dict(model=name,task=task,existing_output=str(out)));todo.remove((name,task));progress=True;continue
   m=models[name];artifact=Path(m['adapter'] or m['model'])
   if m.get('role')!='shared official Base' and not (artifact/'TRAINING_DONE.json').exists():continue
   save('RUNNING_SUPPLEMENT',current=[name,task],failures=failures)
   p=subprocess.run([P,str(R/'scripts/benchmark_5plus5_20261109/evaluate_v2.py'),'--model',name,'--task',task],cwd=R)
   if p.returncode:failures.append(dict(model=name,task=task,exit_code=p.returncode))
   todo.remove((name,task));progress=True;break
  if not progress:
   if time.monotonic()>deadline:raise TimeoutError('Finite supplement wait expired')
   save('WAITING_EXISTING_FORMAL_TRAINING',failures=failures);time.sleep(60)
 save('ALL_TERMINAL_PENDING_RAW_AUDIT',failures=failures,not_proof_of_success=True)
if __name__=='__main__':main()
