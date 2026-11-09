"""Finite manually started, account-wide max-four GPU formal job."""
import argparse, datetime, json, os, subprocess, sys
from pathlib import Path
import yaml
R=Path('/path/to/project')
P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
C=R/'configs/function_aware_formal_20261109'
L=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--language',choices=['sw','ar','ru'],required=True);ap.add_argument('--method',choices=['all','full','kd','random','function'],required=True);ap.add_argument('--worker',action='store_true');ap.add_argument('--stage',choices=['smoke','final']);ap.add_argument('--gpu-uuid');a=ap.parse_args()
    name=a.language+'_'+a.method;root=L/name
    job=json.loads((C/'jobs.json').read_text())[name]
    env=dict(os.environ,FA_METHOD=a.method,FA_LANGUAGE=a.language,FA_DATA_LOCK=job['data_lock'],FA_SOURCE_EXPOSURE=job['source_exposure'],FA_WEIGHTS=str(C/(a.language+'_weights.json')),HF_HOME='/path/to/storage/hf-cache-qwen3',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    if a.worker:
        assert env['CUDA_VISIBLE_DEVICES']==env['AYA_EXPECTED_GPU_UUID']==a.gpu_uuid
        pre=subprocess.run([P,str(R/'scripts/preflight_gpu.py'),'--uuid',a.gpu_uuid,'--max-user-gpus','4','--min-free-mib','21000'],capture_output=True,text=True)
        (root/(a.stage+'-preflight.json')).write_text(pre.stdout+pre.stderr);assert pre.returncode==0
        subprocess.run([P,str(Path(__file__).with_name('train.py')),'--config',str(C/(name+'_'+a.stage+'.yaml')),'--mode','all_layer' if a.method=='all' else 'full_feature'],check=True,env=env)
        return
    root.mkdir(parents=True,exist_ok=False)
    def save(status,**kwargs):
        (root/'state.json').write_text(json.dumps(dict(status=status,language=a.language,method=a.method,stage=stage,pid=os.getpid(),updated_at=datetime.datetime.now().astimezone().isoformat(),**kwargs),indent=2)+'\n')
    sys.path.insert(0,str(R/'scripts/aya_4b_multilang_20261106'));from gpu_slot import run_job
    stage=None
    try:
        for stage in ['smoke','final']:
            if stage=='final':
                cfg=yaml.safe_load((C/(name+'_smoke.yaml')).read_text());done=json.loads((Path(cfg['output_dir'])/'SMOKE_DONE.json').read_text());assert done['steps']==2 and done['exposure']['samples']==32
            run_job(lambda uuid:[P,str(Path(__file__).resolve()),'--language',a.language,'--method',a.method,'--stage',stage,'--worker','--gpu-uuid',uuid],env,root/(stage+'.log'),save)
        save('COMPLETE_FIXED_FINAL400')
    except Exception as exc: save('FAILED',error=repr(exc));raise
if __name__=='__main__':main()
