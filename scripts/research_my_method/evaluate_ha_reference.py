"""Two frozen full Hausa tasks for one completed400-step source model."""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser()
for k in ['project','model','training-record']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--gpu-uuid',required=True);p.add_argument('--name',required=True);a=p.parse_args();r=a.project
py='/local/envs/qwen3-multilingual-20261031/bin/python';base=Path('/local/models/Qwen3-1.7B');taskroot=Path('/local/datasets/multilingual-cl-sequential-20261103/formal-ha-bench-v1')
output=r/'results/research_my_method'/a.name;logs=Path('/local/logs/qwen3-multilingual/sequential-evaluation')/a.name
assert not output.exists() and not logs.exists();output.mkdir(parents=True);logs.mkdir(parents=True);start=time.time()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(name,value):(output/name).write_text(json.dumps(value,indent=2)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu_uuid,HF_HOME='/local/hf-cache-qwen3',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
def run(file,cmd):
 with file.open('x') as f:subprocess.run(list(map(str,cmd)),cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 receipt=json.loads((a.training_record/'COMPLETED.json').read_text());assert receipt['global_step']==400 and receipt['exit_code']==0 and Path(receipt['output']).resolve()==a.model.resolve()
 state=json.loads((a.model/'trainer_state.json').read_text());assert state['global_step']==state['max_steps']==400
 manifest=json.loads((taskroot/'manifest.json').read_text())
 for rel,digest in manifest['files_sha256'].items():assert sha(taskroot/rel)==digest
 for entry in manifest['global_mmlu']['files'].values():assert sha(Path(entry['path']))==entry['sha256']
 stack=json.loads((r/'configs/sft_v1/stack.lock.json').read_text());identity=stack['harness'];assert subprocess.check_output(['git','-C',identity['source'],'rev-parse','HEAD'],text=True).strip()==identity['commit']
 source={str(p):sha(p) for p in [Path(__file__),r/'scripts/audit_harness_accuracy.py',taskroot/'manifest.json']}
 entries=[]
 for task,group,shots,batch in [('ha-belebele','belebele_hau_Latn_sequential_locked','0','4'),('ha-gmmlu','global_mmlu_full_ha_locked','5','1')]:
  name='SST-NoRefresh-'+task;folder=logs/name
  cmd=[str(Path(py).parent/'lm-eval'),'run','--model','hf','--model_args',f'pretrained={a.model},tokenizer={base},dtype=bfloat16,enable_thinking=False','--tasks',group,'--include_path',str(taskroot),'--num_fewshot',shots,'--apply_chat_template','--batch_size',batch,'--device','cuda:0','--output_path',str(folder/'results'),'--log_samples','--seed','42']
  entries.append({'name':name,'task':task,'model':str(a.model),'command':cmd})
 dump('launch.json',{'actual_method':'sst_norefresh','formal_test_used_for_selection':False,'entries':entries,'source_sha256':source,'weights_sha256':{p.name:sha(p) for p in a.model.glob('*.safetensors')},'started_at':datetime.datetime.now().astimezone().isoformat()})
 completed=[]
 for entry in entries:
  assert all(sha(Path(path))==digest for path,digest in source.items())
  folder=logs/entry['name'];folder.mkdir();run(folder/'gpu-preflight.json',[py,r/'scripts/preflight_gpu.py','--uuid',a.gpu_uuid,'--min-free-mib','22000','--max-user-gpus','4','--settle-seconds','30']);run(folder/'eval.log',entry['command'])
  prefix,total,n=('belebele_hau_',900,1) if entry['task']=='ha-belebele' else ('global_mmlu_full_ha_',14042,57)
  run(folder/'accuracy-audit.json',[py,r/'scripts/audit_harness_accuracy.py','--run-dir',folder,'--task-prefix',prefix,'--expected-total',total,'--expected-tasks',n]);completed.append(entry['name']);dump('progress.json',{'completed':completed,'seconds':time.time()-start})
 dump('DONE.json',{'status':'FORMAL_REPORT_COMPLETE','completed':completed,'formal_test_used_for_selection':False,'seconds':time.time()-start})
except BaseException as error:dump('FAILED.json',{'error':repr(error),'seconds':time.time()-start});raise
