"""Use already-frozen synonym features as a supplementary robustness control."""
from common import *
from jsonl_safe import rows
import csv,numpy as np,torch
from safetensors.torch import load_file
def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))
def main():
 assert (R/'records/RELATIONS_DONE.json').exists()
 units=rows(R/'inputs/relations64.jsonl');maps=load_file(str(R/'assets/relation_diagnostic_maps.safetensors'));items=[];summary=[]
 methods=['full2560','historical_body_shared64','english_pca64','random64','cca64_family_filtered','cca64_shuffled']
 for model in ['Base','A']:
  h=load_file(str(R/'assets/diagnostic_states_v2'/model/'relation_states.safetensors'))
  for pos in ['body','boundary']:
   for method in methods:
    prefix=f'Base.{pos}';me=maps[prefix+'.mean_en'].double();ms=maps[prefix+'.mean_sw'].double()
    pe=None if method=='full2560' else maps[f'{prefix}.{method}.en'].double();ps=None if method=='full2560' else maps[f'{prefix}.{method}.sw'].double()
    for u in units:
     def get(key):
      english=key.startswith('en');z=h[f"{u['id']}.{key}.{pos}"].double()-(me if english else ms);p=pe if english else ps
      return z if p is None else z@p
     good,bad=get('en_positive'),get('en_negative');original,paraphrase=get('sw_original'),get('sw_paraphrase')
     m0=cosine(original,good)-cosine(original,bad);mp=cosine(paraphrase,good)-cosine(paraphrase,bad)
     items.append(dict(model=model,readout=pos,method=method,id=u['id'],family=u['family'],original_margin=m0,paraphrase_margin=mp,original_correct=int(m0>0),paraphrase_correct=int(mp>0),same_decision=int((mp>0)==(m0>0)),both_correct=int(m0>0 and mp>0),quality=u['quality']))
 rng=np.random.default_rng(20261116)
 for model in ['Base','A']:
  for pos in ['body','boundary']:
   for method in methods:
    for family in ['sorting','subtraction_order']:
     rows_=[r for r in items if r['model']==model and r['readout']==pos and r['method']==method and r['family']==family];x=np.array([r['paraphrase_correct']-r['original_correct'] for r in rows_]);ci=np.quantile(x[rng.integers(len(x),size=(4000,len(x)))].mean(1),[.025,.975])
     summary.append(dict(model=model,readout=pos,method=method,family=family,n=len(rows_),original_accuracy=np.mean([r['original_correct'] for r in rows_]),paraphrase_accuracy=np.mean([r['paraphrase_correct'] for r in rows_]),same_decision=np.mean([r['same_decision'] for r in rows_]),both_correct=np.mean([r['both_correct'] for r in rows_]),paraphrase_minus_original=x.mean(),difference_low=ci[0],difference_high=ci[1]))
 for name,rs in [('relation_paraphrase_items.csv',items),('relation_paraphrase_summary.csv',summary)]:
  with (R/'analysis'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 write(R/'records/RELATION_PARAPHRASE_DONE.json',{'at':now(),'source':'Already frozen sw_paraphrase fields; no new forward,rank,layer,seed or fit. Fixed Base maps saved in FP32 reused.','limitations':'Supplement only; main2-direction counterfactual score unchanged. Sorting paraphrase reuses Bactrian source; subtraction changes separator only, not syntactic paraphrase. All provisional.'})
if __name__=='__main__':main()
