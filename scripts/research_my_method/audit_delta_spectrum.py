"""Geometric description of frozen effective SW deltas; no functional claims."""
import argparse,json,sys,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--final',type=Path,required=True);p.add_argument('--kind',choices=['full','lora'],required=True);p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
sys.path[:0]=[str(a.project/'scripts/sst'),str(a.project/'scripts/mechanism')]
import torch
from weights import load_pair,validate_pair,unique_entries
from projector import is_target
torch.set_num_threads(4);t=time.monotonic();initial=torch.load(a.initial,map_location='cpu',weights_only=False,mmap=True);base,final=load_pair(a.base,a.final,a.kind);bp,fp=validate_pair(base,final);rows=[];target_energy=free_energy=unconstrained_energy=0.;seen=0
for name,parameter,aliases in unique_entries(final):
 d=parameter.detach()-bp[name].detach();energy=float(d.double().square().sum());seen+=d.numel()
 if is_target(name):
  assert torch.equal(bp[name],initial['references'][name]),'Base mismatch '+name
  projected=initial['bases'][name].project(d);ef=float(projected.double().square().sum());dot=float((d.double()*projected.double()).sum());residual=float((d.double()-projected.double()).square().sum());target_energy+=energy;free_energy+=ef
  row={'name':name,'elements':d.numel(),'delta_l2':energy**.5,'free_energy_fraction':ef/max(energy,1e-60),'protected_or_cross_energy_fraction':residual/max(energy,1e-60),'projector_pythagorean_relative_error':abs(energy-ef-residual)/max(energy,1e-60),'projector_dot_relative_error':abs(dot-ef)/max(energy,1e-60)};del projected
 else:
  unconstrained_energy+=energy;row={'name':name,'elements':d.numel(),'delta_l2':energy**.5,'outside_SST_target':True}
 rows.append(row)
 with (a.output/'progress.json').open('w') as f:json.dump({'completed_unique_tensors':len(rows),'last_parameter':name,'elapsed_seconds':time.monotonic()-t},f)
 del d
result={'label':a.label,'kind':a.kind,'reference':'fixed initial Base r99, all196 matrices, rectangular nullspaces retained','target_delta_squared_norm':target_energy,'target_free_squared_norm':free_energy,'target_free_energy_fraction':free_energy/max(target_energy,1e-60),'outside_target_delta_squared_norm':unconstrained_energy,'outside_target_delta_energy_fraction':unconstrained_energy/max(target_energy+unconstrained_energy,1e-60),'unique_elements':seen,'rows':rows,'seconds':time.monotonic()-t,'evidence_level':'final_parameter_geometry_only','limitations':['Final deltas are not reconstructed optimizer trajectories.','Projecting a trained final delta is not equivalent to SST training.','No causal or benchmark improvement claim follows from spectral energy.','LoRA effective FP32 weights use official PEFT safe merge, not subtraction of factors.']};(a.output/'spectrum.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'DONE.json').write_text(json.dumps({'label':a.label,'target_matrices':sum('outside_SST_target' not in r for r in rows),'seconds':result['seconds']}));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
