"""CPU-only full-weight update norms, with tied embedding/readout checked once."""
from pathlib import Path
from contextlib import ExitStack
import collections,csv,datetime,hashlib,json,re,time
import torch
from safetensors import safe_open
R=Path('/path/to/project');Q=R/'research/fft_representation_20261109';B=Path('/path/to/models/Qwen3-4B')
def load(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def opened(stack,root):
 index=load(root/'model.safetensors.index.json')['weight_map'];files={f:stack.enter_context(safe_open(str(root/f),framework='pt',device='cpu')) for f in set(index.values())};return index,files

def audit(profile,folder):
 done=load(folder/'TRAINING_DONE.json');assert done['steps']==400 and done['exposure']['samples']==6400
 cfg=load(folder/'config.json');assert cfg['tie_word_embeddings'] is True
 assert all(sha(folder/name)==digest for name,digest in done['weights_sha256'].items())
 details=[];groups=collections.defaultdict(float);parameter_count=0;started=time.monotonic()
 with ExitStack() as stack:
  bi,bf=opened(stack,B);ti,tf=opened(stack,folder)
  assert set(ti)-set(bi) <= {'lm_head.weight'} and set(bi)<=set(ti)
  if 'lm_head.weight' in ti:
   embed=tf[ti['model.embed_tokens.weight']].get_slice('model.embed_tokens.weight');head=tf[ti['lm_head.weight']].get_slice('lm_head.weight')
   assert embed.get_shape()==head.get_shape()
   for start in range(0,embed.get_shape()[0],512):assert torch.equal(embed[start:start+512,:].float(),head[start:start+512,:].float()),'Saved shared embedding/head disagree'
  for name in sorted(bi):
   base=bf[bi[name]].get_tensor(name);trained=tf[ti[name]].get_tensor(name);assert base.shape==trained.shape
   flatbase=base.reshape(-1);flattrained=trained.reshape(-1);squared=0.;base_squared=0.
   for start in range(0,base.numel(),1048576):
    x=flatbase[start:start+1048576].float();d=flattrained[start:start+1048576].float()-x
    assert torch.isfinite(d).all();squared+=float(torch.sum(d.double().square()));base_squared+=float(torch.sum(x.double().square()))
   n=base.numel();parameter_count+=n
   if name=='model.embed_tokens.weight':group='shared_embedding_and_readout'
   elif name=='model.norm.weight':group='final_norm'
   elif '.self_attn.' in name:group='decoder_attention'
   elif '.mlp.' in name:group='decoder_mlp'
   else:group='decoder_norm'
   groups[group]+=squared;details.append(dict(profile=profile,parameter=name,group=group,elements=n,update_squared_frobenius=squared,base_squared_frobenius=base_squared))
  assert parameter_count==4022468096
 total=sum(groups.values());return details,dict(profile=profile,unique_parameters=parameter_count,tied_embedding_readout_verified=True,readout_counted_once=True,effective_delta_frobenius=total**.5,groups_squared_frobenius=dict(groups),groups_frobenius={k:v**.5 for k,v in groups.items()},weights_sha256=done['weights_sha256'],cpu_seconds=time.monotonic()-started,claim='Parameter norm is descriptive, not a proxy for downstream effectiveness')

def main():
 torch.set_num_threads(4);profiles=load(R/'configs/fft_representation_20261109/evaluation_profiles.json');reports=[];pending=[]
 lock=load(R/'configs/aya_4b_20261105/model.lock.json')
 assert lock['revision']=='1cfa9a7208912126459214e8b04321603b3df60c'
 assert all(sha(B/name)==info['sha256'] for name,info in lock['files'].items())
 for name in ['historical_fft','historical_ssu','matched_fft','fft_full36']:
  folder=Path(profiles[name]['artifact'])
  if not (folder/'TRAINING_DONE.json').exists():pending.append(name);continue
  cache=Q/(name+'_update_audit.json')
  if cache.exists() and load(cache)['weights_sha256']==load(folder/'TRAINING_DONE.json')['weights_sha256']:
   assert all(sha(folder/name)==digest for name,digest in load(cache)['weights_sha256'].items())
   reports.append(load(cache));continue
  details,report=audit(name,folder);cache.write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
  with (Q/(name+'_parameter_updates.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
 (Q/'full_update_audit.json').write_text(json.dumps(dict(updated_at=datetime.datetime.now().astimezone().isoformat(),status='INCOMPLETE' if pending else 'ALL_FULL_MODEL_UPDATES_AUDITED',models=reports,pending=pending,additional_gpu_jobs=0,base_revision=lock['revision'],base_model_files_rehashed=True),indent=2)+'\n')
 print(json.dumps(dict(complete=len(reports),pending=pending)))
if __name__=='__main__':main()
