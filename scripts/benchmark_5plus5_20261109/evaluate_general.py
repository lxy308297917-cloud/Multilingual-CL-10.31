"""Fixed Final400 evaluation stages, using the existing benchmark implementations."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/path/to/project')
P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python';E='/path/to/storage/envs/qwen3-eval-vllm-20261101/bin/python'
B=Path('/path/to/models/Qwen3-4B')
K=Path('/path/to/storage/checkpoints/function_aware_formal_20261109')
D=Path('/path/to/storage/datasets/function-aware-formal-20261109/evaluation-v1')
L=Path('/path/to/logs/qwen3-multilingual/benchmark-5plus5-20261109')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--stage',choices=['mmlu','code'],required=True);a=ap.parse_args();job=json.loads((R/'configs/benchmark_5plus5_20261109/models.json').read_text())[a.model];a.language=job['language'];a.method='historical_lora'
 lang=a.language;stage=a.stage;name=a.model;adapter=Path(job['adapter'])
 if adapter:
  done=json.loads((adapter/'TRAINING_DONE.json').read_text());assert done['steps']==400 and done['exposure']['samples']==6400;identity=sha(adapter/'adapter_model.safetensors');assert identity==done['weights_sha256']['adapter_model.safetensors']
 else:identity='OFFICIAL_BASE_REVISION_1cfa9a7208912126459214e8b04321603b3df60c'
 out=L/name/stage;out.mkdir(parents=True,exist_ok=False)
 def save(status,**more):(out/'state.json').write_text(json.dumps(dict(status=status,name=name,stage=stage,pid=os.getpid(),updated_at=datetime.datetime.now().astimezone().isoformat(),**more),indent=2)+'\n')
 env=dict(os.environ,HF_HOME='/path/to/storage/hf-cache-qwen3',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NLTK_DATA='/path/to/storage/datasets/qwen3-nltk-data-20261031',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',VLLM_WORKER_MULTIPROC_METHOD='spawn')
 hfargs=f'pretrained={B},tokenizer={B},dtype=bfloat16,enable_thinking=False'+('' if adapter is None else ',peft='+str(adapter))
 def hf(tasks,include,shots=0,batch=4):return [P,'-m','lm_eval','run','--model','hf','--model_args',hfargs,'--tasks',tasks,'--include_path',str(include),'--num_fewshot',str(shots),'--apply_chat_template','--batch_size',str(batch),'--device','cuda:0','--output_path',str(out/'results'),'--log_samples','--seed','42']
 cpu=None
 if stage=='reading':
  target='belebele_swh_Latn_locked' if lang=='sw' else 'aya4bml_belebele_'+lang
  targetxnli='rb_xnli_sw_formal_primary' if lang=='sw' else 'aya4bml_xnli_'+lang
  command=hf(','.join([target,'belebele_eng_Latn_locked',targetxnli,'rb_xnli_en_formal_primary']),D/'tasks')
  builder=lambda uuid:command
 elif stage in ['gmmlu','mmlu']:
  task='mmlu_locked' if stage=='mmlu' else 'global_mmlu_full_'+lang+'_locked'
  include=D/'tasks' if stage=='gmmlu' and lang!='sw' else Path('/path/to/storage/datasets/qwen3-eval-tasks-20261031')
  command=hf(task,include,5,4 if stage=='mmlu' else 1);builder=lambda uuid:command
  cpu=[P,R/'scripts/audit_harness_accuracy.py','--run-dir',out,'--task-prefix','mmlu_' if stage=='mmlu' else 'global_mmlu_full_'+lang+'_','--expected-tasks','57','--expected-total','14042']
 elif stage=='core':
  env.update(VLLM_USE_V1='0',SFT_VLLM_TRACE=str(out/'completion-metadata.jsonl'))
  ma=f'pretrained={B},tokenizer={B},dtype=bfloat16,max_model_len=8192,gpu_memory_utilization=0.65,max_num_seqs=8,max_num_batched_tokens=8192,enable_prefix_caching=True,seed=42,enforce_eager=True,enable_thinking=False'+('' if adapter is None else f',lora_local_path={adapter},max_lora_rank=16')
  command=[E,R/'scripts/sft_v1/harness.py','run','--model','vllm','--model_args',ma,'--tasks','sft_v1_gsm8k,sft_v1_ifeval','--include_path',R/'configs/sft_v1/eval_tasks','--apply_chat_template','--batch_size','auto','--output_path',out/'results','--log_samples','--seed','42'];builder=lambda uuid:command
 else:
  code=stage=='code';env['VLLM_USE_V1']='0' if code else '1'
  dataset=Path('/path/to/storage/datasets/representation-benchmarks-20261109/v1/code') if code else Path('/path/to/storage/datasets/aya-4b-20261105/evaluation-v1/xlsum_sw_v2') if lang=='sw' else Path('/path/to/storage/datasets/aya-4b-multilang-20261106/evaluation-v1')/('xlsum_'+lang+'_v1')
  script=R/'scripts/representation_benchmarks_20261109/generate_code.py' if code else R/'scripts/aya_4b_20261105/generate_v2_gpu_guard.py'
  builder=lambda uuid:[E,script,'--dataset',dataset,'--model',B,'--tokenizer',B,'--method','base' if adapter is None else 'lora',*(() if adapter is None else ['--adapter',adapter]),'--gpu-uuid',uuid,'--output',out/'generation','--scope','full','--batch-size','8','--max-user-gpus','5']
  if code:
   cpu=[P,R/'scripts/math_sft_base_fft/execute_code.py','--generation',out/'generation','--output',out/'execution','--role',name,'--calibration','/path/to/logs/qwen3-multilingual/representation-benchmarks-20261109/formal-base-code/execution']
  elif lang=='sw':cpu=[E,R/'scripts/bench_v2/score.py','--run',out/'generation','--output',out/'scores']
  else:cpu=[P,R/'scripts/aya_4b_multilang_20261106/score_summary_ar_ru.py','--run',out/'generation','--language',lang,'--output',out/'scores']
 sys.path.insert(0,str(R/'scripts/benchmark_5plus5_20261109'));from gpu_slot_five import run_job
 try:
  run_job(builder,env,out/'inference.log',save)
  if cpu:
   save('CPU_SCORING')
   with (out/'scoring.log').open('x') as f:subprocess.run(list(map(str,cpu)),env=dict(env,CUDA_VISIBLE_DEVICES=''),cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
  if adapter:assert sha(adapter/'adapter_model.safetensors')==identity
  (out/'DONE.json').write_text(json.dumps(dict(status='FIXED_ENDPOINT_STAGE_COMPLETE_PENDING_PAIRED_AUDIT',language=lang,method=a.method,stage=stage,model_sha256=identity,no_checkpoint_selection=True),indent=2)+'\n');save('COMPLETE')
 except Exception as exc:save('FAILED',error=repr(exc));raise
if __name__=='__main__':main()
