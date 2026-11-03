"""Extract exact compact modes of acquired SST updates, read-only and CPU-only."""
import argparse,hashlib,json,sys,time,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from coefficient_geometry import compact_update_modes
p=argparse.ArgumentParser();p.add_argument('--adapter-code',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-step',type=int,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);sys.path.insert(0,str(a.adapter_code));torch.set_num_threads(4);t=time.time()
torch.set_grad_enabled(False)
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from projector import is_target

def tsha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
try:
 saved=torch.load(a.checkpoint/'sst-state.pt',map_location='cpu',weights_only=False,mmap=True);assert saved['step']==a.expected_step;assert saved['ratio']==.99;master_hashes=saved['master_sha256'];initial_sha=saved['initial_sha256'];metadata={'step':saved['step'],'interval':saved['interval'],'refresh_steps':saved['refresh_steps']};del saved
 initial=torch.load(a.initial,map_location='cpu',weights_only=False,mmap=True);masters=get_fp32_state_dict_from_zero_checkpoint(str(a.checkpoint),lazy_mode=True);names=sorted(initial['references']);assert len(names)==196 and all(is_target(n) for n in names);banks={};rows=[];total_coeff=0;total_factor_elements=0;sum_energy=0.;max_orth=0.
 for i,n in enumerate(names):
  w=masters[n].contiguous();assert tsha(w)==master_hashes[n],n;reference=initial['references'][n];delta=w-reference;u,s,v=compact_update_modes(delta,initial['bases'][n]);r=len(s);orth_error=float(((u.T@u)*(v.T@v)-torch.eye(r)).abs().max()) if r else 0.;assert orth_error<5e-5,(n,orth_error);max_orth=max(max_orth,orth_error)
  banks[n]={'u':u,'s':s,'v':v};total_coeff+=r;total_factor_elements+=u.numel()+v.numel();energy=float(s.double().square().sum());sum_energy+=energy;rows.append({'name':n,'shape':list(delta.shape),'rank':r,'projected_update_squared_norm':energy,'frobenius_basis_gram_max_absolute_error':orth_error,'master_sha256_verified':True});del w,delta,u,s,v
  if (i+1)%28==0:
   (a.output/'progress.json').write_text(json.dumps({'matrices':i+1,'total':196,'seconds':time.time()-t}));print(json.dumps({'matrices':i+1,'coefficients':total_coeff}),flush=True)
 assert total_coeff<=3556 and total_factor_elements<=21159936;torch.save({'version':1,'source_checkpoint_metadata':metadata,'initial_sha256':initial_sha,'modes':banks,'coefficient_count':total_coeff,'projected_update_squared_norm':sum_energy},a.output/'bank.pt');result={'status':'CPU_BANK_COMPLETE','source_checkpoint':str(a.checkpoint),'source_initial':str(a.initial),'initial_sha256_from_verified_checkpoint':initial_sha,'initial_identity_basis':'Immutable initial artifact already hash-verified by the source training/resume run; all196 source FP32 master hashes rechecked here','source_checkpoint_metadata':metadata,'matrices':196,'coefficients':total_coeff,'factor_elements':total_factor_elements,'factor_bytes':total_factor_elements*4,'projected_update_norm':sum_energy**.5,'max_frobenius_basis_gram_error':max_orth,'relative_SVD_rank_tolerance':1e-7,'decomposition':'Exact SVD of compact fixed-free-space projected update, not randomized/approximate SVD','bank_sha256':hashlib.sha256((a.output/'bank.pt').read_bytes()).hexdigest(),'seconds':time.time()-t,'functional_evidence':False,'rows':rows};(a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
except BaseException as e:(a.output/'FAILED.json').write_text(json.dumps({'error':repr(e),'seconds':time.time()-t}));raise
