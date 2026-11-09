"""One finite frozen evaluation. GPU handoff counts the entire account."""
import argparse, datetime, hashlib, json, os, subprocess, sys, time
from pathlib import Path
R=Path('/data0/users/lixiaoyang/10.31/multilingual-research')
D=Path('/data1/lixiaoyang/datasets/representation-benchmarks-20261109/v1')
P='/data1/lixiaoyang/envs/qwen3-multilingual-20261031/bin/python'
E='/data1/lixiaoyang/envs/qwen3-eval-vllm-20261101/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=['Base','all_layer','full_feature','ce_replay']);ap.add_argument('--stage',required=True,choices=['xnli','code']);ap.add_argument('--scope',required=True,choices=['smoke','formal']);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assets=json.loads((R/'configs/representation_benchmarks_20261109/assets.json').read_text());m=assets['models'][a.model];base=m['base_path'];adapter=m['adapter']
 if adapter:
  assert sha(Path(adapter)/'adapter_model.safetensors')==m['adapter_sha256']
  assert sha(Path(adapter)/'adapter_config.json')==m['adapter_config_sha256']
 a.output.mkdir(parents=True,exist_ok=False)
 def save(status,**more):
  data=dict(status=status,model=a.model,stage=a.stage,scope=a.scope,pid=os.getpid(),updated_at=datetime.datetime.now().astimezone().isoformat(),**more)
  tmp=a.output/'state.json.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(a.output/'state.json')
 env={**os.environ,'HF_HOME':'/data1/lixiaoyang/hf-cache-qwen3','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','VLLM_WORKER_MULTIPROC_METHOD':'spawn','VLLM_USE_V1':'0'}
 if a.stage=='xnli':
  tasks=[x for x in assets['tasks'] if x['scope']==a.scope]
  for x in tasks:
   assert sha(x['data'])==x['data_sha256'];assert sha(D/'tasks'/(x['name']+'.yaml'))==x['task_config_sha256']
  ma=f'pretrained={base},tokenizer={base},dtype=bfloat16,enable_thinking=False'+('' if adapter is None else ',peft='+adapter)
  command=[P,'-m','lm_eval','run','--model','hf','--model_args',ma,'--tasks',','.join(x['name'] for x in tasks),'--include_path',D/'tasks','--num_fewshot','0','--apply_chat_template','--batch_size','4','--device','cuda:0','--log_samples','--seed','42','--output_path',a.output/'results']
  builder=lambda uuid:command
 else:
  # V0 requires a prefill budget at least as large as its maximum sequence.
  builder=lambda uuid:[E,Path(__file__).with_name('generate_code.py'),'--dataset',D/'code','--model',base,'--tokenizer',base,'--method','base' if adapter is None else 'lora',*(() if adapter is None else ('--adapter',adapter)),'--gpu-uuid',uuid,'--output',a.output/'generation','--scope','smoke' if a.scope=='smoke' else 'full','--batch-size','8','--max-user-gpus','4']
 sys.path.insert(0,str(R/'scripts/aya_4b_multilang_20261106'));from gpu_slot import run_job
 start=time.monotonic()
 try:
  run_job(builder,env,a.output/'inference.log',save)
  if adapter:assert sha(Path(adapter)/'adapter_model.safetensors')==m['adapter_sha256']
  report=dict(status='INFERENCE_COMPLETE_PENDING_INDEPENDENT_AUDIT',model=a.model,stage=a.stage,scope=a.scope,elapsed_seconds_including_wait=time.monotonic()-start,model_identity=m)
  (a.output/'DONE.json').write_text(json.dumps(report,indent=2)+'\n');save('COMPLETE')
 except Exception as error:save('FAILED',error=repr(error));raise
if __name__=='__main__':main()
