from pathlib import Path
import json,itertools,hashlib,time,argparse,csv
import numpy as np
import torch
from safetensors.torch import load_file,save_file
R=Path(__file__).resolve().parents[1];OLD=Path('/srv/research/reproductions/shared_subspace_analysis_20261115')
torch.set_num_threads(4)
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def basis(x,r=128):
 _,s,v=torch.linalg.svd(x.double(),full_matrices=False)
 assert s[r-1]>s[0]*1e-8
 return v[:r].T,s
def unit(x):return x/x.norm(dim=1,keepdim=True).clamp_min(1e-12)
def metrics(x,y):
 sim=(unit(x)@unit(y).T).numpy();n=len(sim);ids=np.arange(n)
 # Stable retrieval ties broken by original row order. Also save per-item outcomes.
 rank1=np.argsort(np.argsort(-sim,axis=1,kind='stable'),axis=1,kind='stable')[ids,ids]+1
 rank2=np.argsort(np.argsort(-sim.T,axis=1,kind='stable'),axis=1,kind='stable')[ids,ids]+1
 return dict(en_sw_top1=float(np.mean(rank1==1)),sw_en_top1=float(np.mean(rank2==1)),en_sw_mrr=float(np.mean(1/rank1)),sw_en_mrr=float(np.mean(1/rank2)),en_sw_hits=(rank1==1).astype(int).tolist(),sw_en_hits=(rank2==1).astype(int).tolist(),paired_cos=float(np.diag(sim).mean()),cyclic_mismatch_cos=float(sim[ids,(ids+1)%n].mean()))
def old_decomposition():
 out=R/'analysis/alignment_error_v2.json';assert not out.exists()
 rows=list(map(json.loads,(OLD/'inputs.jsonl').read_text().splitlines()));base=load_file(str(OLD/'states/Base.safetensors'))
 q=load_file('/srv/storage/datasets/s3-bactrian-20261113/teacher-v1/teacher.safetensors')['shared'].double()
 allrows=[]
 for name in ['Base','A','B','S','E','D']:
  states=load_file(str(OLD/'states'/f'{name}.safetensors'))
  for split in ['fit','dev','flores']:
   ix=[i for i,r in enumerate(rows) if r['split']==split]
   teacher=sum(base[f'{l}.L18_body'][ix].double() for l in ['en','zh','fr','de'])/4
   e=states['sw.L18_body'][ix].double()-teacher;proj=e@q
   for part,z in [('shared',proj),('complement',e-proj@q.T),('full',e)]:
    total=float(z.square().sum(1).mean());mean=float(z.mean(0).square().sum());instance=float((z-z.mean(0)).square().sum(1).mean())
    assert abs(total-mean-instance)<max(1e-8,total*1e-10)
    effective_dimension={'shared':64,'complement':2496,'full':2560}[part]
    allrows.append(dict(model=name,split=split,n=len(ix),space=part,total_error=total,mean_error=mean,instance_error=instance,mean_fraction=mean/total,effective_dimension=effective_dimension,per_dimension=total/effective_dimension))
 write(out,dict(status='COMPLETE',readout='historical complete user body mean, raw block18, frozen historical64-d projector and Base four-language teacher',scope='512 fit cached rows,256 exposed dev,128 exposed FLORES dev; no new independent holdout',rows=allrows))
 print('alignment error decomposition complete',flush=True)
def new_analysis(name):
 out=R/'analysis'/f'{name}_geometry.json';assert not out.exists();start=time.monotonic()
 p=json.loads((R/'protocol.json').read_text());states=load_file(str(R/'states'/f'{name}.safetensors'))
 results=[];pairwise=[];stability=[];support=[];saved={};fitn=1024
 for key in ['L18_body']+[f'L{l}_boundary' for l in p['layers']]:
  h={l:states[f'{l}.{key}'][:fitn].double() for l in p['languages']};mu={l:x.mean(0) for l,x in h.items()};x={l:h[l]-mu[l] for l in h}
  qs={};spectra={}
  for l in h:qs[l],spectra[l]=basis(x[l])
  joint,joint_s=basis(torch.cat([qs[l] for l in p['languages'][:4]],1).T/2,64)
  methods={'joint64':joint,'english_pca64':qs['en'][:,:64]}
  for seed in p['random_seeds']:methods[f'random{seed}']=torch.linalg.qr(torch.randn(2560,64,generator=torch.Generator().manual_seed(seed),dtype=torch.float64),mode='reduced')[0]
  for a,b in itertools.combinations(h,2):
   v=torch.linalg.svdvals(qs[a].T@qs[b]);pairwise.append(dict(readout=key,a=a,b=b,overlap128=float(v.square().mean()),angles=torch.rad2deg(torch.acos(v.clamp(0,1))).tolist()))
  for l in h:
   check=states[f'{l}.{key}'][fitn:].double()-mu[l]
   q1,_=basis(x[l][::2]-x[l][::2].mean(0));q2,_=basis(x[l][1::2]-x[l][1::2].mean(0))
   stability.append(dict(readout=key,language=l,split_half_overlap128=float((q1.T@q2).square().sum()/128),pca_gap128=float((spectra[l][127]-spectra[l][128])/spectra[l][127])))
   support.append(dict(readout=key,language=l,support=float((qs[l].T@joint).square().sum()/64),check_energy=float((check@joint).square().sum()/check.square().sum())))
  # All four leave-one-reference-out spaces are reported, never selected by SW outcome.
  for held in p['languages'][:4]:
   leave,_=basis(torch.cat([qs[l] for l in p['languages'][:4] if l!=held],1).T/(3**.5),64)
   for l in [held,'sw']:
    z=states[f'{l}.{key}'][fitn:].double()-mu[l]
    support.append(dict(readout=key,language=l,leftout=held,support=float((qs[l].T@leave).square().sum()/64),check_energy=float((z@leave).square().sum()/z.square().sum())))
  dev={l:states[f'{l}.{key}'][fitn:].double()-mu[l] for l in ['en','sw']}
  for method,q in methods.items():results.append(dict(readout=key,method=method,**metrics(dev['en']@q,dev['sw']@q)))
  results.append(dict(readout=key,method='full2560',**metrics(dev['en'],dev['sw'])))
  # PCA128 + ridge CCA64; fixed ridge=.1 * mean variance, consumes SW pairing.
  z={l:x[l]@qs[l] for l in ['en','sw']};wh={}
  for l in z:
   c=z[l].T@z[l]/(fitn-1);ev,v=torch.linalg.eigh(c+.1*torch.trace(c)/128*torch.eye(128,dtype=torch.float64));wh[l]=(v*ev.rsqrt())@v.T
  for shuffled in [False,True]:
   zy=z['sw'] if not shuffled else z['sw'][torch.randperm(fitn,generator=torch.Generator().manual_seed(1115))]
   u,s,vh=torch.linalg.svd(wh['en']@(z['en'].T@zy/(fitn-1))@wh['sw']);maps={'en':qs['en']@wh['en']@u[:,:64],'sw':qs['sw']@wh['sw']@vh.T[:,:64]}
   results.append(dict(readout=key,method='cca64_shuffled' if shuffled else 'cca64_target_informed',fit_canonical_mean=float(s[:64].mean()),**metrics(dev['en']@maps['en'],dev['sw']@maps['sw'])))
  if key in ['L18_body','L18_boundary']:
   saved.update({f'{key}.{m}':q.float().contiguous() for m,q in methods.items()})
   saved.update({f'{key}.mean_{l}':v.float().contiguous() for l,v in mu.items()})
  print(name,key,round(time.monotonic()-start,1),flush=True)
 write(out,dict(status='COMPLETE',model=name,scores=results,pairwise=pairwise,stability=stability,support=support,seconds=time.monotonic()-start))
 save_file(saved,str(R/'analysis'/f'{name}_bases.safetensors'))
 if name=='Base':
  historical=load_file('/srv/storage/datasets/s3-bactrian-20261113/teacher-v1/teacher.safetensors')
  q=saved['L18_body.joint64'].double();old=historical['shared'].double()
  write(R/'records/historical_reconstruction.json',dict(projector_overlap=float((old.T@q).square().sum()/64),sw_relative_error=float((states['sw.L18_body'][:1024]-historical['base_sw']).norm()/historical['base_sw'].norm())))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['Base','A']);ap.add_argument('--old',action='store_true');a=ap.parse_args()
 if a.old:old_decomposition()
 if a.model:new_analysis(a.model)
