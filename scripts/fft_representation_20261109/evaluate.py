"""Full frozen FFT comparison, including translation and actual HumanEval+ execution."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys
R=Path('/path/to/project');C=R/'configs/fft_representation_20261109'
L=Path('/path/to/logs/qwen3-multilingual/fft-representation-20261109/evaluation')
P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python';E='/path/to/storage/envs/qwen3-eval-vllm-20261101/bin/python'
B=Path('/path/to/models/Qwen3-4B')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',required=True);ap.add_argument('--stage',choices=['reading','summary','gmmlu','mmlu','core','code','mt_en_sw','mt_sw_en'],required=True);a=ap.parse_args()
 profile=json.loads((C/'evaluation_profiles.json').read_text())[a.profile];path=Path(profile['artifact']);kind=profile['kind'];adapter=path if kind=='lora' else None;model=B if adapter else path
 done=json.loads((path/'TRAINING_DONE.json').read_text());assert done['steps']==400 and done['exposure']['samples']==6400
 assert all(sha(path/name)==digest for name,digest in done['weights_sha256'].items())
 identity=done['weights_sha256'];out=L/a.profile/a.stage;out.mkdir(parents=True,exist_ok=False);stage=a.stage
 def save(status,**more):(out/'state.json').write_text(json.dumps(dict(status=status,profile=a.profile,stage=stage,pid=os.getpid(),updated_at=datetime.datetime.now().astimezone().isoformat(),**more),indent=2)+'\n')
 env=dict(os.environ,HF_HOME='/path/to/storage/hf-cache-qwen3',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NLTK_DATA='/path/to/storage/datasets/qwen3-nltk-data-20261031',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',VLLM_WORKER_MULTIPROC_METHOD='spawn')
 hfargs=f'pretrained={model},tokenizer={B},dtype=bfloat16,enable_thinking=False'+('' if adapter is None else ',peft='+str(adapter));cpu=None
 def hf(tasks,include,shots=0,batch=4):return [P,'-m','lm_eval','run','--model','hf','--model_args',hfargs,'--tasks',tasks,'--include_path',include,'--num_fewshot',str(shots),'--apply_chat_template','--batch_size',str(batch),'--device','cuda:0','--output_path',out/'results','--log_samples','--seed','42']
 if stage=='reading':
  tasks='belebele_eng_Latn_locked,belebele_swh_Latn_locked,rb_xnli_sw_formal_primary,rb_xnli_en_formal_primary'
  command=hf(tasks,Path('/path/to/storage/datasets/function-aware-formal-20261109/evaluation-v1/tasks'));builder=lambda uuid:command
 elif stage in ['gmmlu','mmlu']:
  name='global_mmlu_full_sw_locked' if stage=='gmmlu' else 'mmlu_locked';command=hf(name,Path('/path/to/storage/datasets/qwen3-eval-tasks-20261031'),5,1 if stage=='gmmlu' else 4);builder=lambda uuid:command
  cpu=[P,R/'scripts/audit_harness_accuracy.py','--run-dir',out,'--task-prefix','global_mmlu_full_sw_' if stage=='gmmlu' else 'mmlu_','--expected-tasks','57','--expected-total','14042']
 elif stage=='core':
  env.update(VLLM_USE_V1='0',SFT_VLLM_TRACE=str(out/'completion-metadata.jsonl'))
  ma=f'pretrained={model},tokenizer={B},dtype=bfloat16,max_model_len=8192,gpu_memory_utilization=0.65,max_num_seqs=8,max_num_batched_tokens=8192,enable_prefix_caching=True,seed=42,enforce_eager=True,enable_thinking=False'+('' if adapter is None else f',lora_local_path={adapter},max_lora_rank=16')
  command=[E,R/'scripts/sft_v1/harness.py','run','--model','vllm','--model_args',ma,'--tasks','sft_v1_gsm8k,sft_v1_ifeval','--include_path',R/'configs/sft_v1/eval_tasks','--apply_chat_template','--batch_size','auto','--output_path',out/'results','--log_samples','--seed','42'];builder=lambda uuid:command
 else:
  code=stage=='code';env['VLLM_USE_V1']='0' if code else '1'
  name={'summary':'xlsum_sw_v2','mt_en_sw':'flores200_en_sw_full_devtest_v2','mt_sw_en':'flores200_sw_en_full_devtest_v2'}.get(stage)
  dataset=Path('/path/to/storage/datasets/representation-benchmarks-20261109/v1/code') if code else Path('/path/to/storage/datasets/aya-4b-20261105/evaluation-v1')/name
  script=R/'scripts/representation_benchmarks_20261109/generate_code.py' if code else R/'scripts/aya_4b_20261105/generate_v2_gpu_guard.py'
  builder=lambda uuid:[E,script,'--dataset',dataset,'--model',model,'--tokenizer',B,'--method','lora' if adapter else 'full',*(() if adapter is None else ['--adapter',adapter]),'--gpu-uuid',uuid,'--output',out/'generation','--scope','full','--batch-size','8','--max-user-gpus','4']
  cpu=[P,R/'scripts/math_sft_base_fft/execute_code.py','--generation',out/'generation','--output',out/'execution','--role',a.profile,'--calibration','/path/to/logs/qwen3-multilingual/representation-benchmarks-20261109/formal-base-code/execution'] if code else [E,R/'scripts/bench_v2/score.py','--run',out/'generation','--output',out/'scores']
 sys.path.insert(0,str(R/'scripts/aya_4b_multilang_20261106'));from gpu_slot import run_job
 try:
  run_job(builder,env,out/'inference.log',save)
  if cpu:
   save('CPU_SCORING')
   with (out/'scoring.log').open('x') as f:subprocess.run(list(map(str,cpu)),cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
  assert all(sha(path/name)==digest for name,digest in identity.items())
  (out/'DONE.json').write_text(json.dumps(dict(status='FIXED_ENDPOINT_STAGE_COMPLETE_PENDING_PAIRED_AUDIT',profile=a.profile,stage=stage,weights_sha256=identity,endpoint='Final400',no_test_selection=True),indent=2)+'\n');save('COMPLETE')
 except Exception as exc:save('FAILED',error=repr(exc));raise
if __name__=='__main__':main()
