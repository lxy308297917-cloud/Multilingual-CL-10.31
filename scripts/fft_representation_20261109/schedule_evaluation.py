"""Finite actual benchmark queues; reuse complete historical scores."""
import datetime,json,subprocess,time
from pathlib import Path
R=Path('/path/to/project');P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python';L=Path('/path/to/logs/qwen3-multilingual/fft-representation-20261109')
def main():
 profiles=json.loads((R/'configs/fft_representation_20261109/evaluation_profiles.json').read_text())
 pending=[('historical_fft','code'),('historical_ssu','code')]+[(p,s) for p in ['matched_fft','fft_full36'] for s in ['reading','summary','gmmlu','mmlu','core','code','mt_en_sw','mt_sw_en']]+[(p,'mmlu') for p in ['p2_full36','p2_ce']]+[(p,s) for p in ['p2_all','p2_full36','p2_ce'] for s in ['mt_en_sw','mt_sw_en']]
 active={};complete=[];failed=[];deadline=time.monotonic()+120*3600
 while pending or active:
  if time.monotonic()>deadline:raise TimeoutError('Finite120h FFT evaluation queue expired; no retry')
  for task,(child,stream) in list(active.items()):
   if child.poll() is not None:
    stream.close();(complete if child.returncode==0 else failed).append(dict(profile=task[0],stage=task[1],exit_code=child.returncode));del active[task]
  for profile,stage in pending[:]:
   if len(active)>=2:break
   if not (Path(profiles[profile]['artifact'])/'TRAINING_DONE.json').exists():continue
   parent=L/'evaluation-parents';parent.mkdir(exist_ok=True);stream=(parent/(profile+'_'+stage+'.log')).open('x')
   child=subprocess.Popen([P,str(Path(__file__).with_name('evaluate.py')),'--profile',profile,'--stage',stage],cwd=R,stdout=stream,stderr=subprocess.STDOUT)
   active[profile,stage]=(child,stream);pending.remove((profile,stage))
  (L/'evaluation-schedule.json').write_text(json.dumps(dict(status='RUNNING' if pending or active else 'TERMINAL_REQUIRES_AUDIT',pending=pending,active=[dict(profile=k[0],stage=k[1],pid=v[0].pid) for k,v in active.items()],completed=complete,failed=failed,account_gpu_limit=4,updated_at=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
  if pending or active:time.sleep(60)
if __name__=='__main__':main()
