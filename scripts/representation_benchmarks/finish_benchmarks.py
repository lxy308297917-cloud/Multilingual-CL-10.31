"""Finite CPU audit after all four fixed endpoint evaluations have completed."""
from pathlib import Path
import json,subprocess,time,os
R=Path('/data0/users/lixiaoyang/10.31/multilingual-research')
L=Path('/data0/users/lixiaoyang/logs/qwen3-multilingual/representation-benchmarks-20261109')
P='/data1/lixiaoyang/envs/qwen3-multilingual-20261031/bin/python'
def main():
 deadline=time.monotonic()+24*3600
 while True:
  assert time.monotonic()<deadline
  states=[L/('formal-'+m.lower()+'-'+stage)/'DONE.json' for m in ['Base','all_layer','full_feature','ce_replay'] for stage in ['xnli','code']]
  execution=[L/('formal-'+m.lower()+'-code')/'execution/DONE.json' for m in ['Base','all_layer','full_feature','ce_replay']]
  if all(p.exists() for p in states+execution):break
  failures=list(L.glob('formal-*/FAILED.json'))
  assert not failures,failures
  time.sleep(30)
 with (L/'benchmark-analysis.log').open('x') as f:subprocess.run([P,str(R/'scripts/representation_benchmarks_20261109/analyze_benchmarks.py')],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
