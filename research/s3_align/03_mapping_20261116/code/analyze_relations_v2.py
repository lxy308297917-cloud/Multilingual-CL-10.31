from common import *
from jsonl_safe import rows
from prepare_training import cca,pca
from safetensors.torch import load_file,save_file
import numpy as np,torch,csv,itertools
torch.set_num_threads(4)
def cosine(x,y):return float(torch.dot(x,y)/(x.norm()*y.norm()).clamp_min(1e-12))
def csvout(path,rs):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
 units=rows(R/'inputs/relations64.jsonl');fit=read(R/'records/RELATION_DATA_FROZEN.json')['fit_indices']
 allmaps={};saved={}
 for model in ['Base','A']:
  h=load_file(str(PREV/'states'/f'{model}.safetensors'))
  for pos in ['body','boundary']:
   en=h[f'en.L18_{pos}'][fit].double();sw=h[f'sw.L18_{pos}'][fit].double();me,ms=en.mean(0),sw.mean(0);x,y=en-me,sw-ms
   pe,ps,_=cca(x,y);order=torch.randperm(len(fit),generator=torch.Generator().manual_seed(1115));se,ss,_=cca(x,y[order])
   v=load_file(str(R/'assets/mappings.safetensors'))['shared'].double()
   random=torch.linalg.qr(torch.randn(2560,64,generator=torch.Generator().manual_seed(1115),dtype=torch.float64),mode='reduced')[0]
   enpca=pca(x)[:,:64]
   methods={'full2560':(None,None),'historical_body_shared64':(v,v),'english_pca64':(enpca,enpca),'random64':(random,random),'cca64_family_filtered':(pe,ps),'cca64_shuffled':(se,ss)}
   allmaps[model,pos]=(me,ms,methods)
   saved[f'{model}.{pos}.mean_en']=me.float();saved[f'{model}.{pos}.mean_sw']=ms.float()
   for name,(a,b) in methods.items():
    if a is not None:saved[f'{model}.{pos}.{name}.en']=a.float().contiguous();saved[f'{model}.{pos}.{name}.sw']=b.float().contiguous()
 save_file(saved,str(R/'assets/relation_diagnostic_maps.safetensors'))
 # Training-coordinate CCA is a separate fixed-1024 fit, not renamed family-isolated.
 training=load_file(str(R/'assets/mappings.safetensors'));records=[];retrieval=[]
 for model in ['Base','A']:
  states=load_file(str(R/'assets/diagnostic_states_v2'/model/'relation_states.safetensors'))
  for pos in ['body','boundary']:
   for mapmodel in ['Base']+(['A'] if model=='A' else []):
    me,ms,methods=allmaps[mapmodel,pos]
    regimes=[('fixed_Base_family_filtered' if mapmodel=='Base' else 'refitted_A_family_filtered',me,ms,methods)]
    if mapmodel=='Base' and pos=='body':
     actual={f'{arm}_actual_training_standardized':tuple(training[f'{arm}.projection.{lang}'].double()/training[f'{arm}.std.{lang}'].double() for lang in ['en','sw']) for arm in ['T1','T2']}
     regimes.append(('fixed_Base_original1024_training_formula',training['body.mean.en'].double(),training['body.mean.sw'].double(),actual))
    for regime,me,ms,methods in regimes:
     for method,(pe,ps) in methods.items():
      identity=dict(model=model,readout=pos,mapping=regime,method=method)
      es=[];ss=[]
      for u in units:
       def get(k):
        z=states[f"{u['id']}.{k}.{pos}"].double()-(me if k.startswith('en') else ms);p=pe if k.startswith('en') else ps
        return z if p is None else z@p
       positive,negative=get('en_positive'),get('en_negative');es.append(get('en_original'));ss.append(get('sw_original'))
       for direction,key in [('original','sw_original'),('counterfactual','sw_negative')]:
        query=get(key);good,bad=(positive,negative) if direction=='original' else (negative,positive)
        cp,cn=cosine(query,good),cosine(query,bad)
        records.append(dict(**identity,id=u['id'],family=u['family'],split=u['split'],direction=direction,positive_cos=cp,negative_cos=cn,margin=cp-cn,correct=float(cp>cn),tie=float(cp==cn),quality=u['quality']))
      e=torch.stack(es);s=torch.stack(ss);sim=(e/e.norm(dim=1,keepdim=True).clamp_min(1e-12))@(s/s.norm(dim=1,keepdim=True).clamp_min(1e-12)).T
      order=torch.argsort(sim,dim=1,descending=True,stable=True);ranks=torch.argsort(order,dim=1,stable=True)[torch.arange(64),torch.arange(64)]+1
      retrieval.append(dict(**identity,n=64,en_sw_top1=float((ranks==1).double().mean()),en_sw_mrr=float((1/ranks.double()).mean())))
 csvout(R/'analysis/relation_checks.csv',records);csvout(R/'analysis/relation_identity_retrieval.csv',retrieval)
 rng=np.random.default_rng(20261116);summary=[]
 for key in sorted(set((r['model'],r['readout'],r['mapping'],r['method']) for r in records)):
  for family in ['all','sorting','subtraction_order']:
   rs=[r for r in records if (r['model'],r['readout'],r['mapping'],r['method'])==key and (family=='all' or r['family']==family)]
   ids=sorted(set(r['id'] for r in rs));perunit=np.array([[np.mean([r[k] for r in rs if r['id']==i]) for k in ['correct','margin']] for i in ids]);bs=rng.integers(len(ids),size=(4000,len(ids)));ci=np.quantile(perunit[bs].mean(1),[.025,.975],axis=0)
   summary.append(dict(model=key[0],readout=key[1],mapping=key[2],method=key[3],family=family,units=len(ids),pair_order_accuracy=float(perunit[:,0].mean()),accuracy_low=float(ci[0,0]),accuracy_high=float(ci[1,0]),margin=float(perunit[:,1].mean()),margin_low=float(ci[0,1]),margin_high=float(ci[1,1])))
 csvout(R/'analysis/relation_summary.csv',summary)
 paired=[]
 for model,pos,group in itertools.product(['Base','A'],['body','boundary'],['all','sorting','subtraction_order']):
  comparisons=[('fixed_Base_family_filtered','cca64_family_filtered','historical_body_shared64'),('fixed_Base_family_filtered','cca64_family_filtered','full2560'),('fixed_Base_family_filtered','cca64_family_filtered','cca64_shuffled')]
  if pos=='body':comparisons.append(('fixed_Base_original1024_training_formula','T2_actual_training_standardized','T1_actual_training_standardized'))
  for regime,after,before in comparisons:
   rr=[r for r in records if r['model']==model and r['readout']==pos and r['mapping']==regime and (group=='all' or r['family']==group)]
   ids=sorted(set(r['id'] for r in rr));d=[]
   for i in ids:
    d.append([np.mean([r[k] for r in rr if r['id']==i and r['method']==after])-np.mean([r[k] for r in rr if r['id']==i and r['method']==before]) for k in ['correct','margin']])
   d=np.array(d);bs=rng.integers(len(ids),size=(4000,len(ids)));ci=np.quantile(d[bs].mean(1),[.025,.975],axis=0)
   paired.append(dict(model=model,readout=pos,mapping=regime,family=group,after=after,before=before,n_units=len(ids),accuracy_difference=float(d[:,0].mean()),accuracy_low=float(ci[0,0]),accuracy_high=float(ci[1,0]),margin_difference=float(d[:,1].mean()),margin_low=float(ci[0,1]),margin_high=float(ci[1,1])))
 csvout(R/'analysis/relation_paired_intervals.csv',paired)
 failures=[r for r in records if r['method']=='cca64_family_filtered' and r['mapping']=='fixed_Base_family_filtered' and not r['correct']]
 jsonl(R/'analysis/relation_failures.jsonl',[dict(**r,texts=next(u['texts'] for u in units if u['id']==r['id'])) for r in failures])
 write(R/'records/RELATIONS_DONE.json',dict(at=now(),units=64,template_families=2,fit_rows=len(fit),quality='All provisional; mechanically verified answers, no native human review. No natural-language semantic success claim.',uncertainty='Paired bootstrap within units conditional on2templates. Not template-population generalization.',fixed_vs_refit='FixedBase main; refittedA diagnostic only; original1024 training maps separately reported.',historical_shared_boundary='Historical body64 applied to boundary explicitly named; CCA/PCA fit separately per readout.'))
if __name__=='__main__':main()
