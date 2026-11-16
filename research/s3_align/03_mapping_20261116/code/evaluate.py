"""Reuse actual upstream scoring and the previously frozen scientific protocols."""
from common import *
import argparse,os,subprocess
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['Base','T0','T1','T2'],required=True);ap.add_argument('--stage',choices=['reading','core','gmmlu','mmlu'],required=True);a=ap.parse_args()
 lock=read(R/'EVALUATION_LOCK.json')
 for p,d in lock['files'].items():assert sha(p)==d,p
 adapter=None if a.arm=='Base' else CK/f'{a.arm}-full-v1'
 digest=None
 if adapter:
  done=read(adapter/'TRAINING_DONE.json');verified=read(adapter/'RELOAD_VERIFIED.json');assert done['status']=='TRAINING_DONE' and done['steps']==4089 and done['weights_sha256']==verified['weights_sha256']
  digest=sha(adapter/'adapter_model.safetensors');assert digest==done['weights_sha256']['adapter_model.safetensors']
 out=LOG/'evaluation'/a.arm/a.stage;out.mkdir(parents=True,exist_ok=False)
 if a.stage=='core':
  env=dict(os.environ,VLLM_USE_V1='0',SFT_VLLM_TRACE=str(out/'completion-metadata.jsonl'))
  ma=f'pretrained={BASE},tokenizer={BASE},dtype=bfloat16,max_model_len=8192,gpu_memory_utilization=0.82,max_num_seqs=32,max_num_batched_tokens=8192,enable_prefix_caching=True,seed=42,enforce_eager=True,enable_thinking=False'
  if adapter:ma+=f',lora_local_path={adapter},max_lora_rank=16'
  cmd=[EP,str(P/'scripts/sft_v1/harness.py'),'run','--model','vllm','--model_args',ma,'--tasks','sft_v1_gsm8k,sft_v1_ifeval','--include_path',str(P/'configs/sft_v1/eval_tasks'),'--apply_chat_template','--batch_size','auto','--output_path',str(out/'results'),'--log_samples','--seed','42']
 else:
  env=os.environ.copy();ma=f'pretrained={BASE},tokenizer={BASE},dtype=bfloat16,enable_thinking=False,max_batch_size=4'
  if adapter:ma+=f',peft={adapter}'
  include=Path('/srv/storage/datasets/function-aware-formal-20261109/evaluation-v1/tasks') if a.stage=='reading' else Path('/srv/storage/datasets/qwen3-eval-tasks-20261031')
  tasks={'reading':'belebele_swh_Latn_locked','gmmlu':'global_mmlu_full_sw_locked','mmlu':'mmlu_locked'}[a.stage]
  cmd=[TP,'-m','lm_eval','run','--model','hf','--model_args',ma,'--tasks',tasks,'--include_path',str(include),'--num_fewshot','0' if a.stage=='reading' else '5','--apply_chat_template','--batch_size','auto:4','--device','cuda:0','--output_path',str(out/'results'),'--log_samples','--seed','42']
 write(out/'PLAN.json',dict(at=now(),arm=a.arm,stage=a.stage,command=cmd,adapter_sha256=digest,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],evaluation_lock=sha(R/'EVALUATION_LOCK.json'),budget='FULL-65K'))
 result=subprocess.run(cmd,cwd=P,env=env)
 write(out/('DONE.json' if result.returncode==0 else 'FAILED.json'),dict(at=now(),returncode=result.returncode,arm=a.arm,stage=a.stage,adapter_sha256=digest,status='COMPLETE_PENDING_SCORE_AUDIT' if not result.returncode else 'FAILED'))
 raise SystemExit(result.returncode)
if __name__=='__main__':main()
