"""Run only user-prioritized five+five stages; preserve deferred older scope."""
from pathlib import Path
import argparse,datetime,json,subprocess,time
R=Path('/path/to/project');P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python';ROOT=Path('/path/to/logs/qwen3-multilingual')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['fa','fft'],required=True);a=ap.parse_args();plan=json.loads((R/'configs/benchmark_5plus5_20261109/priority_evaluation_plan.json').read_text())[a.family];L=ROOT/plan['log_family'];pending=[tuple(x) for x in plan['pending']];running={};completed=plan['completed'];failed=plan['failed'];deadline=time.monotonic()+120*3600
 while pending or running:
  if time.monotonic()>deadline:raise TimeoutError('Finite priority queue expired')
  for key,(child,stream) in list(running.items()):
   if child.poll() is not None:stream.close();(completed if child.returncode==0 else failed).append(dict(**{plan['key']:key[0]},stage=key[1],exit_code=child.returncode));del running[key]
  for name,stage in pending[:]:
   if len(running)>=plan['workers']:break
   artifact=plan['artifacts'].get(name)
   if artifact and not (Path(artifact)/'TRAINING_DONE.json').exists():continue
   out=L/'evaluation'/name/stage
   if (out/'DONE.json').exists():pending.remove((name,stage));continue
   if out.exists():raise RuntimeError('Preserve existing unfinished attempt, do not overwrite: '+str(out))
   args=['--language',name[:2],'--method',name.split('_',1)[1]] if a.family=='fa' else ['--profile',name]
   parent=L/'priority-evaluation-parents';parent.mkdir(exist_ok=True);stream=(parent/(name+'_'+stage+'.log')).open('x');child=subprocess.Popen([P,str(R/'scripts'/plan['script_family']/'evaluate.py'),*args,'--stage',stage],cwd=R,stdout=stream,stderr=subprocess.STDOUT);running[name,stage]=(child,stream);pending.remove((name,stage))
  d=dict(status='RUNNING' if pending or running else 'AVAILABLE_PRIORITIZED_STAGES_TERMINAL',pending=pending,completed=completed,failed=failed,user_deferred=plan['deferred'],updated_at=datetime.datetime.now().astimezone().isoformat());d['running' if a.family=='fa' else 'active']=[dict(**{plan['key']:k[0]},stage=k[1],pid=v[0].pid) for k,v in running.items()];(L/'evaluation-schedule.json').write_text(json.dumps(d,indent=2)+'\n')
  if pending or running:time.sleep(30)
if __name__=='__main__':main()
