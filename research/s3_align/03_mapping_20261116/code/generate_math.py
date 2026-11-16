from common import *
import argparse,os,subprocess
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['Base','A','B','E'],required=True);a=ap.parse_args()
 modelpath=read(PREV/'protocol.json')['models'][a.model];out=LOG/'math256'/a.model;out.mkdir(parents=True,exist_ok=False)
 if a.model!='Base':assert sha(Path(modelpath)/'adapter_model.safetensors')==read(PREV/'protocol.json')['identities'][a.model]['sha256']
 for p,d in read(R/'records/MATH_DATA_FROZEN.json')['hashes'].items():assert sha(p)==d,p
 ma=f'pretrained={BASE},tokenizer={BASE},dtype=bfloat16,max_model_len=8192,gpu_memory_utilization=0.82,max_num_seqs=32,max_num_batched_tokens=8192,enable_prefix_caching=True,seed=42,enforce_eager=True,enable_thinking=False'
 if a.model!='Base':ma+=f',lora_local_path={modelpath},max_lora_rank=16'
 cmd=[EP,str(P/'scripts/sft_v1/harness.py'),'run','--model','vllm','--model_args',ma,'--tasks','s3map_math256','--include_path',str(R/'inputs/math_task'),'--apply_chat_template','--batch_size','auto','--output_path',str(out/'results'),'--log_samples','--seed','42']
 write(out/'PLAN.json',dict(at=now(),model=a.model,model_path=modelpath,command=cmd,task_sha256=sha(R/'inputs/math_task/math256.yaml'),data_sha256=sha(R/'inputs/math256.parquet'),scope='Training-source new diagnostic256,not formal benchmark',adapter_sha256=None if a.model=='Base' else sha(Path(modelpath)/'adapter_model.safetensors')))
 env=dict(os.environ,VLLM_USE_V1='0',SFT_VLLM_TRACE=str(out/'completion-metadata.jsonl'))
 result=subprocess.run(cmd,env=env,cwd=P)
 write(out/('DONE.json' if result.returncode==0 else 'FAILED.json'),dict(at=now(),model=a.model,returncode=result.returncode));raise SystemExit(result.returncode)
if __name__=='__main__':main()
