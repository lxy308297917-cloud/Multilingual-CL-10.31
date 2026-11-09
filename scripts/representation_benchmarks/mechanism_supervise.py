"""Wait for authorized formal inference, validate a tiny extraction, then four weights."""
from pathlib import Path
import concurrent.futures as cf
import json,subprocess,time,os
R=Path('/data0/users/lixiaoyang/10.31/multilingual-research')
L=Path('/data0/users/lixiaoyang/logs/qwen3-multilingual/representation-benchmarks-20261109')
D=Path('/data1/lixiaoyang/datasets/representation-benchmarks-20261109/v1')
P='/data1/lixiaoyang/envs/qwen3-multilingual-20261031/bin/python'
def run(m,engineering=False):
 name=('engineering-' if engineering else '')+'representations-'+m.lower();cache=D/('features-engineering-v1' if engineering else 'features')/m
 cmd=[P,str(R/'scripts/representation_benchmarks_20261109/extract.py'),'--model',m,'--cache',str(cache),'--output',str(L/name),*(['--engineering'] if engineering else [])]
 with (L/(name+'.launch.log')).open('x') as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 return m
def main():
 deadline=time.monotonic()+24*3600
 def save(status,**more):(L/'mechanism-state.json').write_text(json.dumps(dict(status=status,pid=os.getpid(),new_training_jobs=0,**more),indent=2)+'\n')
 save('WAITING_FORMAL_BENCHMARK_INFERENCE_PRIORITY')
 while True:
  assert time.monotonic()<deadline
  if (L/'series.json').exists():
   s=json.loads((L/'series.json').read_text());assert not s['failed'],s
   if s['status']=='FINITE_BATCH_COMPLETE' and (L/'formal-base-xnli/DONE.json').exists():break
  time.sleep(30)
 save('ENGINEERING_BASE_TWO_OLD_TWO_PAIRS');run('Base',True)
 import torch
 from safetensors.torch import load_file
 torch.set_num_threads(2)
 ref=load_file('/data1/lixiaoyang/datasets/functional-representation-20261109/p1-v1/base_features_v1/base_features.safetensors')['features'];v=load_file(str(D/'features-engineering-v1/Base/old.safetensors'))['features']
 drift=float((v.double()-ref[:len(v)].double()).square().sum()/ref[:len(v)].double().square().sum());assert drift<1e-8,drift
 p=load_file(str(D/'features-engineering-v1/Base/parallel.safetensors'))['pooled'];assert p.shape==(2,2,3,37,2560) and torch.isfinite(p).all()
 # LoRA has no embedding updates: every language/template embedding feature must match Base.
 save('ENGINEERING_LORA_TWO_OLD_TWO_PAIRS');run('full_feature',True)
 q=load_file(str(D/'features-engineering-v1/full_feature/parallel.safetensors'))['pooled'];assert torch.equal(p[:,:,:,0],q[:,:,:,0])
 save('FOUR_FROZEN_MODELS_EXTRACTION',base_teacher_check=drift)
 with cf.ThreadPoolExecutor(max_workers=4) as pool:
  for m in pool.map(run,['Base','all_layer','full_feature','ce_replay']):print('EXTRACTED '+m,flush=True)
 save('CPU_MECHANISM_SUMMARIES')
 with (L/'mechanism-analysis.log').open('x') as f:subprocess.run([P,str(R/'scripts/representation_benchmarks_20261109/analyze_representations.py')],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4'},stdout=f,stderr=subprocess.STDOUT,check=True)
 save('COMPLETE')
if __name__=='__main__':main()
