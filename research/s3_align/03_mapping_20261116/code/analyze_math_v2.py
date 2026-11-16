from common import *
from jsonl_safe import rows
import numpy as np,csv,torch,collections
from safetensors.torch import load_file
def csvout(p,rs):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
 inputs=rows(R/'inputs/math256.jsonl');data={};metadata={};states={};formal={}
 for m in ['Base','A','B','E']:
  out=R/'assets/diagnostic_states_v2'/m;assert read(out/'DONE.json')['math_n']==256
  data[m]={r['id']:r for r in rows(out/'math_reference_probabilities.jsonl')};states[m]=load_file(str(out/'math_states.safetensors'))
  log=LOG/'math256'/m;assert read(log/'DONE.json')['returncode']==0
  files=list((log/'results').rglob('samples_s3map_math256_*.jsonl'));assert len(files)==1
  rr=rows(files[0]);formal[m]={r['doc_id']:r for r in rr if r['filter']=='flexible-extract'};assert len(formal[m])==256
  metadata[m]={'strict':{r['doc_id']:r for r in rr if r['filter']=='strict-match'},'samples_file':str(files[0]),'sha256':sha(files[0])}
 results=[];outputs=[]
 for i,row in enumerate(inputs):
  base=formal['Base'][i];assert base['doc']['question']==row['question']
  assert base['arguments']['gen_args_0']['arg_0']==row['prefix'],'Teacherforcing/generation actual prefix mismatch'
  b,a=bool(base['exact_match']),bool(formal['A'][i]['exact_match']);category={(True,True):'retained',(True,False):'forgotten',(False,True):'acquired',(False,False):'persistent_error'}[b,a]
  for model in ['Base','A','B','E']:
   sample=formal[model][i];assert sample['doc']==base['doc'] and sample['prompt_hash']==base['prompt_hash']
   outputs.append(dict(id=row['id'],model=model,question=row['question'],reference=row['answer'],raw_response=sample['resps'],extracted=sample['filtered_resps'],correct=int(sample['exact_match']),strict_correct=int(metadata[model]['strict'][i]['exact_match']),category_Base_A=category))
   before=data['Base'][row['id']];after=data[model][row['id']];assert before['positions']==after['positions'] and before['target_ids']==after['target_ids']
   lp=np.array(after['target_log_probs']);blp=np.array(before['target_log_probs'])
   for layer in [12,18,24]:
    h=states[model][f"math.{row['id']}.L{layer}"].double();hb=states['Base'][f"math.{row['id']}.L{layer}"].double()
    for pos in range(len(h)):
     results.append(dict(id=row['id'],model=model,layer=layer,position_slot=pos,token_position=before['positions'][pos],category_Base_A=category,generated_correct=int(sample['exact_match']),base_generated_correct=int(base['exact_match']),prompt_tokens=row['prompt_tokens'],answer_tokens=row['answer_tokens'],reference_steps=row['reference_steps'],base_reference_nll=float(-blp[pos]),reference_lp_change=float(lp[pos]-blp[pos]),relative_drift=float((h[pos]-hb[pos]).norm()/hb[pos].norm().clamp_min(1e-12))))
 csvout(R/'analysis/math_reference_drift.csv',results);jsonl(R/'analysis/math256_predictions.jsonl',outputs)
 transitions=[]
 for before,after in [('Base','A'),('Base','B'),('Base','E'),('A','B'),('A','E')]:
  count=collections.Counter()
  for i in range(256):
   x,y=bool(formal[before][i]['exact_match']),bool(formal[after][i]['exact_match']);count[{(True,True):'retained',(True,False):'forgotten',(False,True):'acquired',(False,False):'persistent_error'}[x,y]]+=1
  transitions.append(dict(before=before,after=after,n=256,**{k:count[k] for k in ['retained','forgotten','acquired','persistent_error']}))
 csvout(R/'analysis/math256_transitions.csv',transitions)
 summaries=[]
 for model in ['A','B','E']:
  for layer in [12,18,24]:
   for category in ['retained','forgotten','acquired','persistent_error']:
    rr=[r for r in results if r['model']==model and r['layer']==layer and r['category_Base_A']==category]
    if not rr:continue
    n=len(set(r['id'] for r in rr));means={key:float(np.mean([r[key] for r in rr])) for key in ['relative_drift','reference_lp_change','base_reference_nll','prompt_tokens','answer_tokens','reference_steps']}
    summaries.append(dict(model=model,layer=layer,category_Base_A=category,n=n,**means))
 csvout(R/'analysis/math_group_summary.csv',summaries)
 # Descriptive residual correlation after known length/difficulty covariates. No causal claim.
 adjusted=[]
 for model in ['A','B','E']:
  for layer in [12,18,24]:
   rr=[r for r in results if r['model']==model and r['layer']==layer];per=[]
   for row in inputs:
    t=[r for r in rr if r['id']==row['id']];per.append([np.mean([r[k] for r in t]) for k in ['relative_drift','reference_lp_change','prompt_tokens','answer_tokens','reference_steps','base_reference_nll']])
   x=np.array(per);cov=x[:,2:];cov=(cov-cov.mean(0))/cov.std(0).clip(1e-12);design=np.column_stack([np.ones(256),cov]);res=x[:,:2]-design@np.linalg.lstsq(design,x[:,:2],rcond=None)[0]
   adjusted.append(dict(model=model,layer=layer,n=256,raw_correlation=float(np.corrcoef(x[:,:2].T)[0,1]),length_difficulty_adjusted_correlation=float(np.corrcoef(res.T)[0,1])))
 csvout(R/'analysis/math_confound_checks.csv',adjusted)
 write(R/'records/MATH_DIAGNOSTICS_DONE.json',dict(at=now(),n=256,formal_sources={m:{k:v for k,v in d.items() if k!='strict'} for m,d in metadata.items()},limitations=['Training-source diagnostic,no formaltest tuning','Teacher-forced reference-token probability is not model-generated solution accuracy','Length,reference step count and Base reference NLL are basic covariates,not exhaustive task difficulty','Positions/layers frozen before outputs; all256 retained','Same5shot actualprefix verified for generation and teacherforcing; referenceanswer still added only in teacherforcing']))
if __name__=='__main__':main()
