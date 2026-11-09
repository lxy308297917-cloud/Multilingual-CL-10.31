"""Add missing frozen benchmark scores; never train or change existing runs."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/path/to/project')
P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
B=Path('/path/to/models/Qwen3-4B')
D=Path('/path/to/storage/datasets/benchmark-5plus5-20261109/v1')
L=Path('/path/to/logs/qwen3-multilingual/benchmark-5plus5-20261109')
C=R/'configs/benchmark_5plus5_20261109'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--task',choices=['sib200','gmmlu'],required=True);a=ap.parse_args()
 job=json.loads((C/'models.json').read_text())[a.model];lang=job['language'];model=Path(job['model']);adapter=Path(job['adapter']) if job.get('adapter') else None
 if adapter:
  done=json.loads((adapter/'TRAINING_DONE.json').read_text());assert done['steps']==400
  identity=sha(adapter/'adapter_model.safetensors');assert identity==done['weights_sha256']['adapter_model.safetensors']
 elif model==B:
  identity='OFFICIAL_BASE_REVISION_1cfa9a7208912126459214e8b04321603b3df60c'
 else:
  done=json.loads((model/'TRAINING_DONE.json').read_text());assert done['steps']==400
  full_hashes={name:sha(model/name) for name in done['weights_sha256']};assert full_hashes==done['weights_sha256'];identity=hashlib.sha256(json.dumps(full_hashes,sort_keys=True).encode()).hexdigest()
 out=L/a.model/a.task;out.mkdir(parents=True,exist_ok=False)
 def save(status,**extra):(out/'state.json').write_text(json.dumps(dict(status=status,model=a.model,task=a.task,pid=os.getpid(),at=datetime.datetime.now().astimezone().isoformat(),**extra),indent=2)+'\n')
 if a.task=='sib200':
  manifest=json.loads((D/'manifest.json').read_text());assert sha(D/(lang+'.jsonl'))==manifest['languages'][lang]['jsonl_sha256'];include=D/'tasks';tasks='mcl5_sib200_'+lang;shots=0;batch=4
 else:
  include=Path('/path/to/storage/datasets/function-aware-formal-20261109/evaluation-v1/tasks');tasks='global_mmlu_full_'+lang+'_locked';shots=5;batch=1
 env=dict(os.environ,HF_HOME='/path/to/storage/hf-cache-qwen3',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
 ma=f'pretrained={model},tokenizer={B},dtype=bfloat16,enable_thinking=False'+('' if adapter is None else ',peft='+str(adapter))
 command=[P,'-m','lm_eval','run','--model','hf','--model_args',ma,'--tasks',tasks,'--include_path',str(include),'--num_fewshot',str(shots),'--apply_chat_template','--batch_size',str(batch),'--device','cuda:0','--output_path',str(out/'results'),'--log_samples','--seed','42']
 (out/'protocol.json').write_text(json.dumps(dict(model=job,model_sha256=identity,command=command,account_max_gpus=5,temporary_user_authorization='2026-11-09: fifth GPU for benchmark extension',started_at=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
 from gpu_slot_five import run_job
 started=time.monotonic()
 try:
  run_job(lambda uuid:command,env,out/'inference.log',save)
  if a.task=='gmmlu':
   with (out/'audit.log').open('x') as f:subprocess.run([P,str(R/'scripts/audit_harness_accuracy.py'),'--run-dir',str(out),'--task-prefix','global_mmlu_full_'+lang+'_','--expected-tasks','57','--expected-total','14042'],env={**env,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT,check=True,cwd=R)
  if adapter:assert sha(adapter/'adapter_model.safetensors')==identity
  elif model!=B:assert {name:sha(model/name) for name in full_hashes}==full_hashes
  (out/'DONE.json').write_text(json.dumps(dict(status='COMPLETE_PENDING_RAW_PAIRED_AUDIT',model=a.model,task=a.task,model_sha256=identity,elapsed_including_gpu_queue_seconds=time.monotonic()-started),indent=2)+'\n');save('COMPLETE')
 except Exception as e:save('FAILED',error=repr(e));raise
if __name__=='__main__':main()
