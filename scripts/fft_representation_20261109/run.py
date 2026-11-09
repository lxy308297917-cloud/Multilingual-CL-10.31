"""Finite manually queued engineering check followed by complete formal FFT."""
from pathlib import Path
import argparse,datetime,json,os,sys,subprocess
import yaml
R=Path('/path/to/project');C=R/'configs/fft_representation_20261109';L=Path('/path/to/logs/qwen3-multilingual/fft-representation-20261109');P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['none','full36'],required=True);ap.add_argument('--worker',action='store_true');ap.add_argument('--stage',choices=['smoke','final']);ap.add_argument('--gpu-uuid');a=ap.parse_args();root=L/a.mode
 env=dict(os.environ,HF_HOME='/path/to/storage/hf-cache-qwen3',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',TORCH_EXTENSIONS_DIR='/path/to/storage/torch-extensions-qwen3')
 if a.worker:
  assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['AYA_EXPECTED_GPU_UUID']==a.gpu_uuid
  pre=subprocess.run([P,str(R/'scripts/preflight_gpu.py'),'--uuid',a.gpu_uuid,'--max-user-gpus','4','--min-free-mib','21000'],capture_output=True,text=True)
  (root/(a.stage+'-preflight.json')).write_text(pre.stdout+pre.stderr);assert pre.returncode==0
  subprocess.run([P,'-m','torch.distributed.run','--nproc_per_node=1','--standalone',str(Path(__file__).with_name('train.py')),'--config',str(C/(a.mode+'_'+a.stage+'.yaml')),'--mode',a.mode],env=env,check=True);return
 root.mkdir(parents=True,exist_ok=False);stage=None
 sys.path.insert(0,str(R/'scripts/aya_4b_multilang_20261106'));from gpu_slot import run_job
 def save(status,**more):(root/'state.json').write_text(json.dumps(dict(status=status,mode=a.mode,stage=stage,pid=os.getpid(),updated_at=datetime.datetime.now().astimezone().isoformat(),**more),indent=2)+'\n')
 try:
  for stage in ['smoke','final']:
   if stage=='final':
    cfg=yaml.safe_load((C/(a.mode+'_smoke.yaml')).read_text());done=json.loads((Path(cfg['output_dir'])/'SMOKE_DONE.json').read_text());assert done['steps']==3 and done['exposure']['samples']==48 and done['applied_cpuadam_updates']==3 and done['backward_calls']==48
   run_job(lambda uuid:[P,str(Path(__file__).resolve()),'--mode',a.mode,'--stage',stage,'--worker','--gpu-uuid',uuid],env,root/(stage+'.log'),save)
  save('COMPLETE_FIXED_FINAL400')
 except Exception as exc:save('FAILED',error=repr(exc));raise
if __name__=='__main__':main()
