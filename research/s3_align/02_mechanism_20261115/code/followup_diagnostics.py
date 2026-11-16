"""Reproduction of CPU follow-up calculations first run interactively in this study.
Run in a copied/new experiment directory; existing results are never overwritten.
"""
from pathlib import Path
import json,torch,numpy as np
from safetensors.torch import load_file
from analyze import basis,metrics
from summarize import ci
R=Path(__file__).resolve().parents[1]
def save(name,value):
 p=R/'analysis'/name;assert not p.exists(),p;p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def main():
 states={m:load_file(str(R/'states'/f'{m}.safetensors')) for m in ['Base','A']}
 bases={m:load_file(str(R/'analysis'/f'{m}_bases.safetensors')) for m in states}
 rotation=[];fixed=[]
 for key in ['L18_body','L18_boundary']:
  row=dict(readout=key,base_A_joint64_overlap=float((bases['Base'][key+'.joint64'].T@bases['A'][key+'.joint64']).square().sum()/64))
  for m in states:
   halves=[]
   for parity in [0,1]:
    qs=[]
    for l in ['en','zh','fr','de']:
     x=states[m][l+'.'+key][:1024][parity::2].double();q,_=basis(x-x.mean(0));qs.append(q)
    q,_=basis(torch.cat(qs,1).T/2,64);halves.append(q)
   row[m+'_joint64_split_half_overlap']=float((halves[0].T@halves[1]).square().sum()/64)
   for center in ['own_fit_mean','base_fit_mean']:
    who=m if center=='own_fit_mean' else 'Base';h={l:states[m][l+'.'+key][1024:]-states[who][l+'.'+key][:1024].mean(0) for l in ['en','sw']};q=bases['Base'][key+'.joint64'];fixed.append(dict(model=m,readout=key,centering=center,**metrics(h['en']@q,h['sw']@q)))
  rotation.append(row)
 save('rotation_stability.json',rotation);save('fixed_base_new.json',fixed)
 comparisons=[]
 for m in ['Base','A']:
  g=json.loads((R/'analysis'/f'{m}_geometry.json').read_text())
  for key in ['L18_body']+[f'L{i}_boundary' for i in [6,12,18,24,30,36]]:
   scores={x['method']:x for x in g['scores'] if x['readout']==key}
   for method in ['english_pca64','full2560','cca64_target_informed','cca64_shuffled']:
    for direction in ['en_sw','sw_en']:
     v=np.array(scores[method][direction+'_hits'])-np.array(scores['joint64'][direction+'_hits']);comparisons.append(dict(model=m,readout=key,direction=direction,method=method,reference='joint64',difference_pp=float(100*v.mean()),paired_ci_pp=[100*x for x in ci(v)]))
 save('retrieval_comparisons.json',comparisons)
 rows=list(map(json.loads,(R/'inputs/intervention.jsonl').read_text().splitlines()))
 baselines={m:{(x['id'],x['language']):x for x in json.loads((R/'records'/f'{m}_task_baselines.json').read_text())} for m in ['Base','A']}
 h={m:load_file(str(R/'states'/f'{m}_task_boundary.safetensors'))['en'].double() for m in baselines}
 q=bases['Base']['L18_boundary.joint64'].double();d=h['A']-h['Base'];inside=(d@q).square().sum(1);total=d.square().sum(1);relative=total/h['Base'].square().sum(1);groups={}
 for i,row in enumerate(rows):
  c0=baselines['Base'][row['id'],'en']['pred']==row['gold'];c1=baselines['A'][row['id'],'en']['pred']==row['gold'];label=('Base对' if c0 else 'Base错')+'→'+('A对' if c1 else 'A错');groups.setdefault(label,[]).append(i)
 result=[dict(group=k,n=len(ix),mean_relative_squared_drift=float(relative[ix].mean()),mean_shared_fraction=float((inside[ix]/total[ix]).mean()),ids=[rows[i]['id'] for i in ix]) for k,ix in groups.items()]
 save('english_task_drift.json',dict(scope='Descriptive same-input L18 actual answer-boundary drift,128 exposed reading passages; no causal claim or length/difficulty-adjusted inference',rows=result))
if __name__=='__main__':main()
