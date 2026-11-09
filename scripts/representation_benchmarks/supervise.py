"""Finite benchmark batch; stops submitting after a failure, no retries/self-start."""
from pathlib import Path
import concurrent.futures as cf
import json,subprocess,time,os
R=Path('/data0/users/lixiaoyang/10.31/multilingual-research')
L=Path('/data0/users/lixiaoyang/logs/qwen3-multilingual/representation-benchmarks-20261109')
P='/data1/lixiaoyang/envs/qwen3-multilingual-20261031/bin/python'
def run(model,stage,scope):
 name=scope+'-'+model.lower()+'-'+stage
 command=[P,str(R/'scripts/representation_benchmarks_20261109/benchmark.py'),'--model',model,'--stage',stage,'--scope',scope,'--output',str(L/name)]
 with (L/(name+'.supervisor.log')).open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
 return name
def main():
 assert (L/'smoke-base-xnli/DONE.json').exists() and (L/'smoke-feature-code/DONE.json').exists()
 run('full_feature','xnli','smoke')
 jobs=[(m,'xnli','formal') for m in ['all_layer','full_feature','ce_replay']]+[(m,'code','formal') for m in ['Base','all_layer','full_feature','ce_replay']]
 state=dict(pid=os.getpid(),new_training_jobs=0,finite_jobs=jobs,completed=[],failed=[],status='RUNNING')
 def save():
  t=L/'series.json.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(L/'series.json')
 save();waiting=iter(jobs)
 with cf.ThreadPoolExecutor(max_workers=4) as pool:
  active={pool.submit(run,*next(waiting)) for _ in range(min(4,len(jobs)))}
  while active:
   done,_=cf.wait(active,return_when=cf.FIRST_COMPLETED)
   for f in done:
    active.remove(f)
    try:state['completed'].append(f.result())
    except Exception as error:state['failed'].append(repr(error))
    if not state['failed']:
     job=next(waiting,None)
     if job is not None:active.add(pool.submit(run,*job))
   save()
 state['status']='FAILED' if state['failed'] else 'FINITE_BATCH_COMPLETE';save()
 if state['failed']:raise RuntimeError(state['failed'])
if __name__=='__main__':main()
