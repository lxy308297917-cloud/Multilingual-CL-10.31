"""Finite manual schedule. No scientific gate, retries, startup hook or score selection."""
import datetime,json,os,subprocess,time
from pathlib import Path
R=Path('/path/to/project')
C=R/'configs/function_aware_formal_20261109'
L=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109')
P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
def main():
 jobs=json.loads((C/'jobs.json').read_text());pending={k:v for k,v in jobs.items() if k not in ['sw_kd','ar_all','ru_all']};children={};events=[];deadline=time.monotonic()+48*3600
 while pending or children:
  if time.monotonic()>deadline:raise TimeoutError('Finite48h launch schedule expired; no retry')
  for name,job in list(pending.items()):
   method=job['method'];lang=job['language']
   if method in ['full','function','random'] and not Path('/path/to/storage/datasets/function-aware-formal-20261109/features-v1/manifest.json').exists():continue
   if method in ['function','random'] and not (C/(lang+'_weights.json')).exists():continue
   out=L/(name+'-coordinator.log');assert not out.exists()
   stream=out.open('x');cmd=[P,str(Path(__file__).with_name('run.py')),'--language',lang,'--method',method]
   child=subprocess.Popen(cmd,cwd=R,stdout=stream,stderr=subprocess.STDOUT);children[name]=(child,stream);del pending[name]
   events.append(dict(name=name,event='QUEUED_ENGINEERING_THEN_FIXED400',pid=child.pid,at=datetime.datetime.now().astimezone().isoformat()))
  for name,(child,stream) in list(children.items()):
   if child.poll() is not None:
    stream.close();events.append(dict(name=name,event='EXIT',code=child.returncode,at=datetime.datetime.now().astimezone().isoformat()));del children[name]
  (L/'schedule.json').write_text(json.dumps(dict(status='RUNNING' if pending or children else 'FINISHED_CHECK_EACH_JOB',pending=list(pending),children={n:x[0].pid for n,x in children.items()},events=events,updated_at=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
  if pending or children:time.sleep(60)
if __name__=='__main__':main()
