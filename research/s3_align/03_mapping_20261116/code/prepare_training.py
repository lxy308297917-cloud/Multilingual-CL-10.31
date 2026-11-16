from common import *
import torch, shutil, random, yaml
from safetensors.torch import load_file,save_file
torch.set_num_threads(4)
def pca(x):return torch.linalg.svd(x,full_matrices=False)[2][:128].T
def cca(x,y):
 qx,qy=pca(x),pca(y);zx,zy=x@qx,y@qy
 def white(z):
  c=z.T@z/(len(z)-1);e,v=torch.linalg.eigh(c+.1*c.trace()/128*torch.eye(128,dtype=torch.float64))
  return (v*e.rsqrt())@v.T
 wx,wy=white(zx),white(zy);u,s,v=torch.linalg.svd(wx@(zx.T@zy/(len(x)-1))@wy)
 return qx@wx@u[:,:64],qy@wy@v.T[:,:64],s
def main():
 assert not (R/'assets/mappings.safetensors').exists()
 source=W/'work/s3-bactrian-20261113/prepared/alignment.jsonl'
 pool=rows(source);representation=rows(PREV/'inputs/representation.jsonl')
 assert len(pool)==1024 and [r['id'] for r in pool]==[r['id'] for r in representation[:1024]]
 shutil.copy2(source,R/'inputs/alignment.jsonl')
 h=load_file(str(PREV/'states/Base.safetensors'));oldteacher=Path('/srv/storage/datasets/s3-bactrian-20261113/teacher-v1/teacher.safetensors')
 v=load_file(str(oldteacher))['shared'].double();tensors={};metadata={}
 for pos in ['body','boundary']:
  en=h[f'en.L18_{pos}'][:1024].double();sw=h[f'sw.L18_{pos}'][:1024].double();me,ms=en.mean(0),sw.mean(0);x,y=en-me,sw-ms
  pe,ps,s=cca(x,y)
  g=torch.Generator().manual_seed(1115);order=torch.randperm(1024,generator=g);se,ss,_=cca(x,y[order])
  for method,maps in [('cca',(pe,ps)),('shuffled_cca',(se,ss)),('english_pca',(pca(x)[:,:64],)*2)]:
   for lang,m in zip(['en','sw'],maps):tensors[f'{pos}.{method}.{lang}']=m.float().contiguous()
  for lang,m in [('en',me),('sw',ms)]:tensors[f'{pos}.mean.{lang}']=m.float()
  metadata[pos]={'canonical_correlations':s[:64].tolist(),'build':1024,'pca_rank':128,'ridge_relative':.1,'output_rank':64}
  if pos=='body':
   for arm,(ae,ass) in [('T1',(v,v)),('T2',(pe,ps))]:
    for lang,projection,data in [('en',ae,x),('sw',ass,y)]:
     std=(data@projection).std(0,correction=1).clamp_min(1e-6)
     tensors[f'{arm}.projection.{lang}']=projection.float().contiguous();tensors[f'{arm}.std.{lang}']=std.float()
    tensors[f'{arm}.teacher']=((x@ae)/tensors[f'{arm}.std.en'].double()).float().contiguous()
 tensors['shared']=v.float().contiguous()
 save_file(tensors,str(R/'assets/mappings.safetensors'))
 keep_path=W/'work/s3-align-v0-20261113/keep_audit/KEEP_CONTRACT.json';keep=read(keep_path)
 calibration_indices=random.Random(42).sample(range(1024),32)
 # CE uses exactly these same bilingual source IDs, with the true SW assistant answer.
 from datasets import load_from_disk
 dataset=load_from_disk(str(DATA/'sw_train_tokenized'))['train'];raw=rows(DATA/'sw_train.jsonl')
 assert len(dataset)==len(raw)==65415
 print('raw example fields',list(raw[0]),'tokenized fields',dataset.column_names,flush=True)
 # Source-to-tokenized row order is validated by the existing manifest and lengths below.
 source_ids=read(DATA/'ids.json')
 write(R/'records/raw_ids_schema.json',{'keys':list(source_ids) if isinstance(source_ids,dict) else None,'first_row':raw[0],'dataset_columns':dataset.column_names})
 write(R/'assets/mapping_metadata.json',dict(at=now(),mapping=metadata,build_ids=[r['id'] for r in pool],calibration_indices=calibration_indices,calibration_ids=[pool[i]['id'] for i in calibration_indices],teacher='Frozen Base English only; existing Base batch1 cache',std_correction=1,std_floor=1e-6,mu=1.,keep_contract=str(keep_path),keep_normalizers=keep['loss']['normalizers'],keep_exposure='one fixed English row per optimizer step, same order as original B/S/E/D',old_8k_scale_difference='Nov15 anchor study recalibrated keep scales; those scales are not silently substituted, historical T0 is rerun.',hashes={str(p):sha(p) for p in [source,PREV/'states/Base.safetensors',oldteacher,keep_path,R/'assets/mappings.safetensors',DATA/'manifest.json',DATA/'ids.json']}))
 base=yaml.safe_load((W/'work/s3-bactrian-20261113/configs/B.yaml').read_text())
 for arm in ['T0','T1','T2']:
  cfg=dict(base,per_device_train_batch_size=2,gradient_accumulation_steps=8,logging_steps=20,save_steps=1000,save_total_limit=1,save_only_model=False,output_dir=str(CK/f'{arm}-full-v1'),logging_dir=str(LOG/f'training/{arm}-full-v1'))
  (R/'configs'/f'{arm}-full-v1.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
  cfg.update(max_steps=8,save_strategy='no',output_dir=str(CK/f'{arm}-smoke-v1'),logging_dir=str(LOG/f'training/{arm}-smoke-v1'))
  (R/'configs'/f'{arm}-smoke-v1.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 print('Mappings and candidate configs prepared',flush=True)
if __name__=='__main__':main()
