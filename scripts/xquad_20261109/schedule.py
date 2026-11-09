"""One finite XQuAD supplement worker; no model training or holding GPUs."""
from pathlib import Path
import datetime,json,os,subprocess,time
R=Path('/path/to/project');L=Path('/path/to/logs/qwen3-multilingual/xquad-20261109');P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
def main():
 L.mkdir(parents=True,exist_ok=True);models=json.loads((R/'configs/xquad_20261109/models.json').read_text());pending=list(models);failed=[];deadline=time.monotonic()+120*3600
 def save(status,**kw):(L/'schedule.json').write_text(json.dumps(dict(status=status,at=datetime.datetime.now().astimezone().isoformat(),pid=os.getpid(),pending=pending,failed=failed,**kw),indent=2)+'\n')
 while pending:
  moved=False
  for name in pending[:]:
   m=models[name];adapter=m.get('adapter')
   if adapter and not (Path(adapter)/'TRAINING_DONE.json').exists():continue
   if (L/name/'DONE.json').exists():pending.remove(name);moved=True;continue
   if (L/name).exists():raise RuntimeError('Existing unfinished XQuAD attempt; preserve and diagnose independently')
   save('RUNNING_OR_WAITING_GPU',current=name);p=subprocess.run([P,str(Path(__file__).with_name('evaluate.py')),'--model',name],cwd=R);pending.remove(name);moved=True
   if p.returncode:failed.append(dict(model=name,exit_code=p.returncode))
   break
  if not moved:
   if time.monotonic()>deadline:raise TimeoutError('Finite XQuAD wait expired')
   save('WAITING_ALREADY_RUNNING_TRAINING');time.sleep(60)
 save('ALL_TERMINAL_PENDING_RAW_AUDIT')
if __name__=='__main__':main()
