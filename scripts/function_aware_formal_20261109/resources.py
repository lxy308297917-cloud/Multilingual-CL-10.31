"""One finite shared reference cache, common to every representation method."""
import argparse,datetime,hashlib,json,os,sys,subprocess,time
from pathlib import Path
R=Path('/path/to/project')
P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
D=Path('/path/to/storage/datasets/functional-representation-20261109/p2-v1')
O=Path('/path/to/storage/datasets/function-aware-formal-20261109/features-v1')
L=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109/features-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--gpu-uuid');a=ap.parse_args()
 if not a.worker:
  O.mkdir(parents=True,exist_ok=False);L.mkdir(parents=True,exist_ok=False)
  sys.path.insert(0,str(R/'scripts/aya_4b_multilang_20261106'));from gpu_slot import run_job
  def save(status,**kwargs):(L/'state.json').write_text(json.dumps(dict(status=status,updated_at=datetime.datetime.now().astimezone().isoformat(),**kwargs),indent=2)+'\n')
  env=dict(os.environ,HF_HOME='/path/to/storage/hf-cache-qwen3',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4')
  try:run_job(lambda uuid:[P,str(Path(__file__).resolve()),'--worker','--gpu-uuid',uuid],env,L/'worker.log',save);save('COMPLETE')
  except Exception as exc:save('FAILED',error=repr(exc));raise
  return
 assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['AYA_EXPECTED_GPU_UUID']==a.gpu_uuid
 import torch
 from transformers import AutoModelForCausalLM
 from peft import PeftModel
 from safetensors.torch import load_file,save_file
 from losses import collect_states
 torch.set_num_threads(4)
 assert str(torch.cuda.get_device_properties(0).uuid)==a.gpu_uuid.removeprefix('GPU-')
 rows=[json.loads(x) for x in (D/'build_pool.jsonl').open()];assert len(rows)==128 and all(x['split']=='build' for x in rows)
 model=AutoModelForCausalLM.from_pretrained('/path/to/models/Qwen3-4B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
 old=load_file(str(D/'build_reference_features.safetensors'));cache={};started=time.monotonic();oracle=0.
 with torch.inference_mode():
  for i,row in enumerate(rows):
   states=collect_states(model,row).cpu();error=float((states[:,1::2]-old[row['id']].float()).abs().max());oracle=max(oracle,error)
   assert error<1e-3,(row['id'],error)
   cache[row['id']]=states.to(torch.bfloat16).contiguous()
   if (i+1)%16==0:print('REFERENCE',i+1,128,flush=True)
  source=Path('/path/to/storage/checkpoints/aya_4b_20261105/lora_final_v1')
  model=PeftModel.from_pretrained(model,source,is_trainable=False).eval().requires_grad_(False)
  values=[]
  for i,row in enumerate(rows):
   current=collect_states(model,row);ref=cache[row['id']].to('cuda').float()
   per=(current-ref).square().sum((0,2))/ref.square().sum((0,2)).clamp_min(1e-12)
   values.append(dict(id=row['id'],capability=row['capability'],normalized_drift=float(per.mean())))
   if (i+1)%16==0:print('SCALE',i+1,128,flush=True)
 scales={cap:sum(x['normalized_drift'] for x in values if x['capability']==cap)/64 for cap in ['math','instruction']};assert min(scales.values())>1e-8
 save_file(cache,str(O/'build_reference_features.safetensors'));(O/'build_pool.jsonl').write_bytes((D/'build_pool.jsonl').read_bytes())
 (O/'normalizer_rows.json').write_text(json.dumps(values,indent=2)+'\n')
 manifest=dict(status='COMMON72_REFERENCE_COMPLETE',files={p.name:sha(p) for p in O.iterdir() if p.is_file()},normalizers=scales,sites=72,groups=6,sites_per_group=12,old36_site_oracle_max_error=oracle,normalizer_reference_adapter_sha256=sha(source/'adapter_model.safetensors'),normalizer_common_across_languages=True,old_pool_sha256=sha(D/'build_pool.jsonl'),check_training_rows=0,elapsed_seconds=time.monotonic()-started)
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
