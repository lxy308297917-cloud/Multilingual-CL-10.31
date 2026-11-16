"""Six-model cache-only semantic diagnostics. No evaluation-set centering."""
from common import *
import torch,numpy as np,csv,itertools,time
from safetensors.torch import load_file
torch.set_num_threads(4)
def unit(x):return x/x.norm(dim=1,keepdim=True).clamp_min(1e-12)
def cka(x,y):
 x=x-x.mean(0);y=y-y.mean(0);a=x@x.T;b=y@y.T
 return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-12))
def metrics(en,sw):
 sim=(unit(en)@unit(sw).T).numpy();n=len(sim);idx=np.arange(n);diag=np.diag(sim)
 result={};items=[]
 for direction,s in [('en_sw',sim),('sw_en',sim.T)]:
  order=np.argsort(-s,axis=1,kind='stable');ranks=np.argsort(order,axis=1,kind='stable')[idx,idx]+1
  other=s.copy();np.fill_diagonal(other,-np.inf)
  gap=diag-(s.sum(1)-diag)/(n-1);hard=diag-other.max(1)
  result.update({direction+'_top1':float(np.mean(ranks==1)),direction+'_mrr':float(np.mean(1/ranks)),direction+'_pair_gap':float(gap.mean()),direction+'_hard_gap':float(hard.mean())})
  items.extend(dict(direction=direction,index=int(i),rank=int(ranks[i]),top_index=int(order[i,0]),paired_cos=float(diag[i]),pair_gap=float(gap[i]),hard_gap=float(hard[i])) for i in idx)
 result['linear_cka']=cka(en,sw)
 return result,items
def csvout(p,rs):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
 assert not (R/'analysis/semantic_comparison.csv').exists()
 oldrows=rows(OLD/'inputs.jsonl');fit=[i for i,r in enumerate(oldrows) if r['split']=='fit'];dev=[i for i,r in enumerate(oldrows) if r['split']=='dev'];assert len(fit)==512 and len(dev)==256
 base=load_file(str(OLD/'states/Base.safetensors'));teacher=load_file('/srv/storage/datasets/s3-bactrian-20261113/teacher-v1/teacher.safetensors')
 qe=torch.linalg.svd(base['en.L18_body'][fit].double()-base['en.L18_body'][fit].double().mean(0),full_matrices=False)[2][:64].T
 methods={'historical_shared64':teacher['shared'].double(),'historical_random64':teacher['random'].double(),'base_english_pca64':qe,'full2560':None}
 summary=[];records=[];errors=read(PREV/'analysis/alignment_error_v2.json')['rows']
 for model in ['Base','A','B','S','E','D']:
  states=load_file(str(OLD/'states'/f'{model}.safetensors'))
  for reference in ['fixed_Base_English_and_means','within_model_own_build_means']:
   ensource=base if reference.startswith('fixed') else states
   mean_source=base if reference.startswith('fixed') else states
   en=ensource['en.L18_body'][dev].double()-ensource['en.L18_body'][fit].double().mean(0)
   sw=states['sw.L18_body'][dev].double()-mean_source['sw.L18_body'][fit].double().mean(0)
   for method,q in methods.items():
    e,s=(en,sw) if q is None else (en@q,sw@q)
    m,its=metrics(e,s);identity=dict(model=model,readout='L18_body',reference=reference,space=method,n=256,build_mean_n=512)
    summary.append(dict(**identity,**m));records.extend(dict(**identity,id=oldrows[dev[i['index']]]['id'],**i) for i in its)
  print(model,'cached body comparison complete',flush=True)
 # Existing actual boundary states are available for Base/A only; no last-body proxy substitution.
 newer={m:load_file(str(PREV/'states'/f'{m}.safetensors')) for m in ['Base','A']}
 bv=load_file(str(PREV/'analysis/Base_bases.safetensors'))
 bmethods={'base_shared64':bv['L18_boundary.joint64'].double(),'base_random64':bv['L18_boundary.random1115'].double(),'base_english_pca64':bv['L18_boundary.english_pca64'].double(),'full2560':None}
 for model,states in newer.items():
  for reference in ['fixed_Base_English_and_means','within_model_own_build_means']:
   es=newer['Base'] if reference.startswith('fixed') else states;ms=newer['Base'] if reference.startswith('fixed') else states
   en=es['en.L18_boundary'][1024:].double()-es['en.L18_boundary'][:1024].double().mean(0)
   sw=states['sw.L18_boundary'][1024:].double()-ms['sw.L18_boundary'][:1024].double().mean(0)
   for method,q in bmethods.items():
    m,its=metrics(en if q is None else en@q,sw if q is None else sw@q)
    identity=dict(model=model,readout='L18_boundary',reference=reference,space=method,n=256,build_mean_n=1024)
    summary.append(dict(**identity,**m));records.extend(dict(**identity,id=oldrows[dev[i['index']]]['id'],**i) for i in its)
 csvout(R/'analysis/semantic_comparison.csv',summary);jsonl(R/'analysis/retrieval_items.jsonl',records)
 joined=[]
 for r in summary:
  if r['readout']=='L18_body' and r['space']=='historical_shared64':
   error=next(x for x in errors if x['model']==r['model'] and x['split']=='dev' and x['space']=='shared')
   joined.append(dict(**r,old_fourlanguage_total_error=error['total_error'],old_fourlanguage_mean_error=error['mean_error'],old_fourlanguage_instance_error=error['instance_error']))
 csvout(R/'analysis/error_vs_retrieval.csv',joined)
 rng=np.random.default_rng(20261116);bootstrap=rng.integers(0,256,size=(4000,256));comparisons=[]
 for after,before in [('S','A'),('E','B'),('E','D')]:
  for ref,method,direction in itertools.product(['fixed_Base_English_and_means','within_model_own_build_means'],methods,['en_sw','sw_en']):
   sets=[]
   for model in [after,before]:
    its=[x for x in records if x['model']==model and x['readout']=='L18_body' and x['reference']==ref and x['space']==method and x['direction']==direction]
    assert len(its)==256;sets.append(its)
   assert [r['id'] for r in sets[0]]==[r['id'] for r in sets[1]]
   for metric in ['top1','mrr','pair_gap','hard_gap']:
    def value(r):return float(r['rank']==1) if metric=='top1' else 1/r['rank'] if metric=='mrr' else r[metric]
    delta=np.array([value(a)-value(b) for a,b in zip(*sets)]);interval=np.quantile(delta[bootstrap].mean(1),[.025,.975])
    comparisons.append(dict(after=after,before=before,reference=ref,space=method,direction=direction,metric=metric,difference=float(delta.mean()),low=float(interval[0]),high=float(interval[1])))
 csvout(R/'analysis/semantic_paired_intervals.csv',comparisons)
 write(R/'records/SEMANTIC_DONE.json',dict(at=now(),n=256,body_models=6,boundary_models=['Base','A'],boundary_other_models='NOT_RUN: no actual prediction-boundary cache; body task is required priority',centering='build only; fixed Base or each own model explicitly separated',cka='Gram centering is intrinsic metric computation, not fitted retrieval centering',bootstrap='4000 source-ID paired; unadjusted exploratory intervals, no cross-seed inference',source_hashes={str(OLD/'states'/f'{m}.safetensors'):sha(OLD/'states'/f'{m}.safetensors') for m in ['Base','A','B','S','E','D']}))
if __name__=='__main__':main()
