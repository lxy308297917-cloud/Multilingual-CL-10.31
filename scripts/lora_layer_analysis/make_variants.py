"""Recreate the six post-training LoRA ablations; no training and no CUDA required.

Validated export of the experimental tensor operations. Original adapters are read
only. This does not reproduce Last-k training, and update magnitude is not a
measure of functional importance.
"""
import argparse,csv,hashlib,json,re,shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file
MODULES=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
VARIANTS=['remove_early','remove_middle','remove_late','keep_late','remove_attention','remove_mlp']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
 return h.hexdigest()
def disabled(variant,layer,module,groups):
 if variant.startswith('remove_') and variant[7:] in groups:return layer in groups[variant[7:]]
 if variant=='keep_late':return layer not in groups['late']
 if variant=='remove_attention':return module in MODULES[:4]
 if variant=='remove_mlp':return module in MODULES[4:]
 raise ValueError(variant)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--adapter',action='append',required=True,help='language=/path/to/original/adapter (repeatable)');p.add_argument('--output',type=Path,required=True);p.add_argument('--norms',action='store_true');a=p.parse_args();torch.set_num_threads(4)
 n=json.loads((a.base/'config.json').read_text())['num_hidden_layers'];groups={'early':list(range(n//3)),'middle':list(range(n//3,2*n//3)),'late':list(range(2*n//3,n))};a.output.mkdir(parents=True,exist_ok=False)
 sources={};records=[];norms=[];base_norm={};index=json.loads((a.base/'model.safetensors.index.json').read_text())['weight_map'] if a.norms else None
 for item in a.adapter:
  lang,directory=item.split('=',1);assert re.fullmatch(r'[a-z][a-z0-9_-]*',lang) and lang not in sources;source=Path(directory);before={name:sha(source/name) for name in ['adapter_model.safetensors','adapter_config.json']};cfg=json.loads((source/'adapter_config.json').read_text());assert cfg['bias']=='none' and not cfg.get('modules_to_save') and not cfg.get('use_dora') and not cfg.get('use_rslora') and not cfg.get('rank_pattern') and not cfg.get('alpha_pattern') and not cfg.get('fan_in_fan_out')
  scale=cfg['lora_alpha']/cfg['r'];weights=load_file(str(source/'adapter_model.safetensors'),device='cpu');akeys=sorted(k for k in weights if k.endswith('lora_A.weight'));assert len(akeys)==n*7 and len(weights)==n*14;infos=[]
  for ak in akeys:
   layer=int(re.search(r'\.layers\.(\d+)\.',ak)[1]);module=ak.split('.')[-3];bk=ak.replace('lora_A','lora_B');assert module in MODULES and 0<=layer<n;count=weights[ak].numel()+weights[bk].numel();infos.append((ak,bk,layer,module,count))
   if a.norms:
    aa=weights[ak].double();bb=weights[bk].double();effective=float((((bb.T@bb)*(aa@aa.T).T).sum().clamp_min(0)).sqrt())*scale;key=ak.removeprefix('base_model.model.').replace('.lora_A.weight','.weight')
    if key not in base_norm:
     with safe_open(str(a.base/index[key]),framework='pt',device='cpu') as f:weight=f.get_tensor(key)
     base_norm[key]=float(torch.linalg.vector_norm(weight.float()));del weight
    norms.append(dict(language=lang,layer=layer,module=module,kind='attention' if module in MODULES[:4] else 'mlp',group=next(k for k,v in groups.items() if layer in v),delta_frobenius=effective,base_frobenius=base_norm[key],relative_frobenius=effective/base_norm[key],adapter_parameters=count,weight_elements=aa.shape[1]*bb.shape[0]))
  for variant in VARIANTS:
   target=a.output/lang/variant;target.mkdir(parents=True,exist_ok=False);new={k:v.clone() for k,v in weights.items()};removed=[];active=0
   for ak,bk,layer,module,count in infos:
    if disabled(variant,layer,module,groups):new[bk].zero_();removed.append(bk)
    else:active+=count
   save_file(new,str(target/'adapter_model.safetensors'),metadata={'format':'pt'});shutil.copy2(source/'adapter_config.json',target/'adapter_config.json');check=load_file(str(target/'adapter_model.safetensors'))
   for key,value in weights.items():
    if key in removed:assert torch.count_nonzero(check[key]).item()==0
    else:assert torch.equal(check[key],value)
   records.append(dict(language=lang,variant=variant,active_adapter_parameters=active,disabled_modules=len(removed),sha256=sha(target/'adapter_model.safetensors'),source_sha256=before,zero_delta_verified=True,training_performed=False))
  assert before=={name:sha(source/name) for name in before};sources[lang]=before
 if norms:
  with (a.output/'module_norms.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(norms[0]));w.writeheader();w.writerows(norms)
 (a.output/'recreation_manifest.json').write_text(json.dumps(dict(layers=n,groups=groups,sources=sources,variants=records),indent=2)+'\n');print(json.dumps({'languages':list(sources),'verified_variants':len(records),'no_training':True}))
if __name__=='__main__':main()
