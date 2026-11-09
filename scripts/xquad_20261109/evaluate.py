"""Fixed Final400 XQuAD generation through standard LM Evaluation Harness."""
import datetime,hashlib,json,os,subprocess,sys,time,argparse
from pathlib import Path
R=Path('/path/to/project');C=R/'configs/xquad_20261109';D=Path('/path/to/storage/datasets/xquad-supplement-20261109/v1');L=Path('/path/to/logs/qwen3-multilingual/xquad-20261109');P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python';B=Path('/path/to/models/Qwen3-4B')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);a=ap.parse_args();m=json.loads((C/'models.json').read_text())[a.model];lang=m['language'];assert lang in ['ar','ru'];adapter=Path(m['adapter']) if m['adapter'] else None
 if adapter:
  done=json.loads((adapter/'TRAINING_DONE.json').read_text());assert done['steps']==400;identity=sha(adapter/'adapter_model.safetensors');assert identity==done['weights_sha256']['adapter_model.safetensors']
 else:identity='OFFICIAL_BASE_REVISION_1cfa9a7208912126459214e8b04321603b3df60c'
 manifest=json.loads((D/'manifest.json').read_text());assert sha(D/(lang+'.jsonl'))==manifest['languages'][lang]['jsonl_sha256'];out=L/a.model;out.mkdir(parents=True,exist_ok=False)
 def save(status,**kw):(out/'state.json').write_text(json.dumps(dict(status=status,model=a.model,pid=os.getpid(),at=datetime.datetime.now().astimezone().isoformat(),**kw),indent=2)+'\n')
 env=dict(os.environ,HF_HOME='/path/to/storage/hf-cache-qwen3',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4');ma=f'pretrained={B},tokenizer={B},dtype=bfloat16,enable_thinking=False'+('' if adapter is None else ',peft='+str(adapter))
 command=[P,'-m','lm_eval','run','--model','hf','--model_args',ma,'--tasks','mcl_xquad_'+lang,'--include_path',str(D/'tasks'),'--num_fewshot','0','--apply_chat_template','--batch_size','4','--device','cuda:0','--output_path',str(out/'results'),'--log_samples','--seed','42'];(out/'protocol.json').write_text(json.dumps(dict(model=m,model_sha256=identity,command=command,data_manifest_sha256=sha(D/'manifest.json'),dataset_sha256=sha(D/(lang+'.jsonl')),started_at=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
 sys.path.insert(0,str(R/'scripts/benchmark_5plus5_20261109'));from gpu_slot_five import run_job
 started=time.monotonic()
 try:
  run_job(lambda uuid:command,env,out/'inference.log',save)
  if adapter:assert sha(adapter/'adapter_model.safetensors')==identity
  (out/'DONE.json').write_text(json.dumps(dict(status='COMPLETE_PENDING_RAW_PAIRED_AUDIT',model=a.model,model_sha256=identity,samples=1190,elapsed_including_queue_seconds=time.monotonic()-started),indent=2)+'\n');save('COMPLETE')
 except Exception as e:save('FAILED',error=repr(e));raise
if __name__=='__main__':main()
