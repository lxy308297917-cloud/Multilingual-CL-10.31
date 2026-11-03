"""Paired four-model HA comparison; separate initial headroom and final learning."""
from pathlib import Path
import argparse,json,hashlib,collections
import numpy as np
from analyze_curves import interval

def point_contrasts(v):
 a0,a1,b0,b1=[np.asarray(v[k],float) for k in ['A0','A1','B0','B1']];assert a0.shape==a1.shape==b0.shape==b1.shape
 return {'A_gain':a1-a0,'B_gain':b1-b0,'initial_B_minus_A':b0-a0,'final_A_minus_B':a1-b1,'A_gain_minus_B_gain':(a1-a0)-(b1-b0)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--development',type=Path,required=True);a=p.parse_args();r=a.project;assert not a.output.exists();a.output.mkdir(parents=True);sources={}
 def read(f):sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();return json.loads(f.read_text())
 def load(record):
  assert (record/'DONE.json').exists() and not (record/'FAILED.json').exists();assert not read(record/'DONE.json')['formal_test_used_for_selection'];out={};launch=read(record/'launch.json')
  for e in launch['entries']:
   if e['task'] not in ['ha-belebele','ha-gmmlu']:continue
   cmd=e['command'];d=Path(cmd[cmd.index('--output_path')+1]).parent;proof=read(d/'accuracy-audit.json');assert proof['status']=='validated';files=list(d.glob('results/**/results_*.json'));assert len(files)==1;summary=read(files[0]);rows={}
   for name in summary['results']:
    fs=list(files[0].parent.glob('samples_'+name+'_*.jsonl'))
    if not fs:continue
    assert len(fs)==1;sources[str(fs[0])]=hashlib.sha256(fs[0].read_bytes()).hexdigest()
    for v in map(json.loads,fs[0].open()):
     assert v['filter']=='none' and v['acc'] in [0,1];key=(name,str(v['doc_id']));assert key not in rows;rows[key]=v
   out[e['task']]=rows
  return out
 records={'A0':r/'results/research_my_method/sequential-fft-formal-v1-source-ha-Base','A1':r/'results/next_stage/ha-only-formal-v1-ha','B0':r/'results/research_my_method/sequential-fft-formal-v1-source-ha-SW-FFT','B1':r/'results/research_my_method/sequential-fft-formal-v1-main'};models={k:load(v) for k,v in records.items()};dev={tuple(x) for x in json.loads((a.development/'reading_ids.json').read_text())};assert len(dev)==180;results=[]
 for task,n in [('ha-belebele',900),('ha-gmmlu',14042)]:
  reference=models['A0'][task];keys=sorted(reference);assert len(keys)==n
  for label in ['A1','B0','B1']:
   assert models[label][task].keys()==reference.keys()
   for key in keys:
    b=reference[key];t=models[label][task][key];assert b['doc']==t['doc'] and b['arguments']==t['arguments'],(task,key,'doc/prompt differs')
  subsets={'all':keys}
  if task=='ha-belebele':
   readingkey=lambda k:(reference[k]['doc']['link'],str(reference[k]['doc']['question_number']))
   subsets['development_overlap180']=[k for k in keys if readingkey(k) in dev];subsets['not_in_current_curves720']=[k for k in keys if readingkey(k) not in dev];assert len(subsets['development_overlap180'])==180 and len(subsets['not_in_current_curves720'])==720
  for subset,selected in subsets.items():
   values={label:np.array([models[label][task][k]['acc'] for k in selected],dtype=float) for label in records};clusters=[reference[k]['doc']['link'] if task=='ha-belebele' else '|'.join(k) for k in selected];contrasts=point_contrasts(values);points={label:100*float(v.mean()) for label,v in values.items()};pairs={}
   for name,delta in contrasts.items():
    z=interval(delta,np.ones(len(delta)),clusters);pairs[name]={k:(v*100 if k in ['delta','ci95_low','ci95_high'] else v) for k,v in z.items()}
   assert abs(pairs['A_gain_minus_B_gain']['delta']-pairs['final_A_minus_B']['delta']-pairs['initial_B_minus_A']['delta'])<1e-9
   strata=[]
   for x,y in [(0,0),(0,1),(1,0),(1,1)]:
    mask=(values['A0']==x)&(values['B0']==y);ii=np.flatnonzero(mask)
    entry={'initial_A0_correct':x,'initial_B0_correct':y,'n':len(ii)}
    if len(ii):
     entry.update(A1_final_acc=100*float(values['A1'][ii].mean()),B1_final_acc=100*float(values['B1'][ii].mean()));z=interval(values['A1'][ii]-values['B1'][ii],np.ones(len(ii)),[clusters[i] for i in ii]);entry['final_A_minus_B']={k:(v*100 if k in ['delta','ci95_low','ci95_high'] else v) for k,v in z.items()}
    strata.append(entry)
   results.append({'task':task,'subset':subset,'n':len(selected),'scores':points,'paired_contrasts_pp':pairs,'predefined_initial_correctness_strata':strata,'cluster_unit':'original_passage' if task=='ha-belebele' else 'original_question','historical_exposure':'All reading900 historically exposed;180 used for current diagnostics.720 not used in current curves but not historically unseen.' if task=='ha-belebele' else 'Official test distinct from dev;prior-model scores historically exposed;new run fixedFinal only'})
 payload={'status':'VERIFIED_FOUR_MODEL_CONTROL','model_roles':{'A0':'BaseW0','A1':'Base→HA400','B0':'SWFFTFinalW1','B1':'SWFFT→HA400'},'results':results,'all_doc_and_prompt_exact':True,'sources_sha256':sources,'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'bootstrap':'2000 paired original clusters,seed42;unadjusted descriptive intervals','no_test_based_model_selection':True,'interpretation_limits':['Gain contrast includes initial headroom;report final gap and initial gap separately','Correctness strata descriptive,not randomized causal proof','No test-based LR/rank/budget/checkpoint change','HA-only total400 versus sequentialtotal800;onlyHA-phase exposures matched']};(a.output/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
 lines=['# Hausa-only与顺序FFT：正确起点/终点及配对对照','','| Task/subset | n | Base A0 | HA-only A1 | SW FFT B0 | SW→HA B1 | A gain | B gain | Final A−B |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for x in results:
  s=x['scores'];c=x['paired_contrasts_pp'];lines.append('| %s/%s | %d | %.2f | %.2f | %.2f | %.2f | %+.2f | %+.2f | %+.2f |'%(x['task'],x['subset'],x['n'],s['A0'],s['A1'],s['B0'],s['B1'],c['A_gain']['delta'],c['B_gain']['delta'],c['final_A_minus_B']['delta']))
 lines+=['','全部区间、初始差、收益差和预定义00/01/10/11条件组见analysis.json。收益差含起点headroom，不单独当作负向迁移证明。阅读720仅与本轮曲线不重叠，不称历史未见。'];(a.output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'comparisons':len(results),'all_doc_and_prompt_exact':True}))
if __name__=='__main__':main()
