from pathlib import Path
import json,csv,collections,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def ci(x):
 x=np.asarray(x,dtype=float);rng=np.random.default_rng(20261115)
 return np.quantile(x[rng.integers(0,len(x),(4000,len(x)))].mean(1),[.025,.975]).tolist()
def main():
 data=[]
 for m in ['Base','A']:
  assert (R/'analysis'/f'intervention_{m}_DONE.json').exists()
  data.extend(map(json.loads,(R/'analysis'/f'intervention_{m}.jsonl').read_text().splitlines()))
 assert len(data)==9600
 groups=collections.defaultdict(list)
 for x in data:groups[x['model'],x['mode'],x['condition'],x['alpha']].append(x)
 for key,rows in groups.items():assert len(rows)==128 and len({r['id'] for r in rows})==128
 baselines={m:{(r['id'],r['language']):r for r in json.loads((R/'records'/f'{m}_task_baselines.json').read_text())} for m in ['Base','A']}
 summaries=[];paired=[];examples=[]
 for model,mode in [('Base','cross_language'),('A','cross_language'),('A','english_restore')]:
  zero=sorted(groups[model,mode,'none',0.],key=lambda r:r['id']);ids=[r['id'] for r in zero];z=np.array([r['acc'] for r in zero]);zm=np.array([r['margin'] for r in zero]);vectors={}
  source=baselines[model if mode=='cross_language' else 'Base'];source_correct=np.array([source[i,'en']['pred']==source[i,'en']['gold'] for i in ids]);repair=source_correct&(z==0)
  for (m,mode2,condition,alpha),rs in groups.items():
   if (m,mode2)!=(model,mode):continue
   rs=sorted(rs,key=lambda r:r['id']);assert [r['id'] for r in rs]==ids
   acc=np.array([r['acc'] for r in rs]);margin=np.array([r['margin'] for r in rs]);vectors[condition,alpha]=acc
   summaries.append(dict(model=model,mode=mode,condition=condition,alpha=alpha,n=128,acc=acc.mean(),delta=100*(acc-z).mean(),delta_ci=[100*x for x in ci(acc-z)],recovered=int(((z==0)&(acc==1)).sum()),damaged=int(((z==1)&(acc==0)).sum()),source_correct_recipient_wrong_n=int(repair.sum()),repair_subset_fixed=int((acc[repair]==1).sum()),mean_margin=margin.mean(),margin_delta=float((margin-zm).mean()),margin_delta_ci=ci(margin-zm),mean_injection_norm=float(np.mean([r['injection_norm'] for r in rs])),clipped=sum(r['scale_clipped'] for r in rs),fallback_rows=sum(r['single_batch_fallback'] for r in rs),top_choice_ties=sum(r['logprobs'].count(max(r['logprobs']))>1 for r in rs)))
  core=vectors['shared',.5]
  for control in [('none',0.),('centered_shared',.5),('unrelated_same_label',.5),('historical_mean_space',.5),('full',.5)]+[(f'random{s}_equalnorm',.5) for s in range(1115,1120)]:
   diff=core-vectors[control];paired.append(dict(model=model,mode=mode,comparison='shared0.5 minus '+str(control),delta=100*diff.mean(),ci=[100*x for x in ci(diff)]))
  avg=np.mean([vectors[f'random{s}_equalnorm',.5] for s in range(1115,1120)],axis=0);diff=core-avg
  paired.append(dict(model=model,mode=mode,comparison='shared0.5 minus mean_five_equalnorm_random',delta=100*diff.mean(),ci=[100*x for x in ci(diff)]))
  changed=sorted(groups[model,mode,'shared',.5],key=lambda r:r['id'])
  for category in ['recovered','damaged','unchanged_wrong']:
   candidates=[r for r,zr in zip(changed,zero) if (category=='recovered' and zr['acc']==0 and r['acc']==1) or (category=='damaged' and zr['acc']==1 and r['acc']==0) or (category=='unchanged_wrong' and zr['acc']==0 and r['acc']==0)]
   for r in candidates[:2]:examples.append(dict(model=model,mode=mode,category=category,id=r['id']))
 write(R/'analysis/intervention_summary.json',dict(status='COMPLETE',summary=summaries,paired=paired,examples=examples,bootstrap='4000 paired resamples by128 distinct passages; conditional on selected data and five fixed random bases, no multiplicity or training-seed adjustment'))
 with (R/'analysis/intervention_summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
 retrieval=[]
 for m in ['Base','A']:
  a=json.loads((R/'analysis'/f'{m}_geometry.json').read_text())
  for s in a['scores']:
   retrieval.append(dict(model=m,**{k:v for k,v in s.items() if not isinstance(v,list)}))
 with (R/'analysis/retrieval.csv').open('w') as f:
  keys=list(dict.fromkeys(k for s in retrieval for k in s));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(retrieval)
 for row in summaries:
  if row['alpha']==.5 and row['condition'] in ['shared','centered_shared','mean_only','unrelated_same_label','full','historical_mean_space','random1115_equalnorm']:print(row['model'],row['mode'],row['condition'],round(row['acc']*100,2),round(row['delta'],2),row['delta_ci'],flush=True)
if __name__=='__main__':main()
