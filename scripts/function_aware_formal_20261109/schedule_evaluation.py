"""Queue formal results as each Final400 completes; no test-driven feedback into training."""
import datetime,json,os,subprocess,time
from pathlib import Path
R=Path('/path/to/project');K=Path('/path/to/storage/checkpoints/function_aware_formal_20261109');P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python';L=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109')
def main():
 jobs=json.loads((R/'configs/function_aware_formal_20261109/jobs.json').read_text())
 pending=[(x,s) for x in jobs for s in ['reading','summary','gmmlu','mmlu','core','code']]+[('sw_all','mmlu'),('sw_Base','mmlu'),('ar_Base','gmmlu'),('ru_Base','gmmlu')]
 running={};completed=[];failed=[];deadline=time.monotonic()+120*3600
 while pending or running:
  if time.monotonic()>deadline:raise TimeoutError('Finite120h formal-evaluation window expired, no retries')
  for key,(child,stream) in list(running.items()):
   if child.poll() is not None:
    stream.close();(completed if child.returncode==0 else failed).append(dict(name=key[0],stage=key[1],exit_code=child.returncode));del running[key]
  for name,stage in pending[:]:
   if len(running)>=4:break
   if name not in ['sw_all','sw_Base','ar_Base','ru_Base'] and not (K/(name+'_final_v1')/'TRAINING_DONE.json').exists():continue
   lang,method=name.split('_',1);log=L/'evaluation-parents';log.mkdir(exist_ok=True)
   stream=(log/(name+'_'+stage+'.log')).open('x');child=subprocess.Popen([P,str(Path(__file__).with_name('evaluate.py')),'--language',lang,'--method',method,'--stage',stage],cwd=R,stdout=stream,stderr=subprocess.STDOUT)
   running[(name,stage)]=(child,stream);pending.remove((name,stage))
  (L/'evaluation-schedule.json').write_text(json.dumps(dict(status='RUNNING' if pending or running else 'FINISHED_CHECK_FAILURES',pending=pending,running=[dict(name=k[0],stage=k[1],pid=v[0].pid) for k,v in running.items()],completed=completed,failed=failed,updated_at=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
  if pending or running:time.sleep(60)
if __name__=='__main__':main()
