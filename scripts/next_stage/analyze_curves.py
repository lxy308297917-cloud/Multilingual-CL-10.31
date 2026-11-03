"""Read all frozen stage points, report all metrics; no checkpoint selection."""
from pathlib import Path
import argparse,json,csv,hashlib,collections
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def interval(delta,counts,clusters,repeats=2000):
 delta=np.asarray(delta,float);counts=np.asarray(counts,float);by=collections.defaultdict(list)
 for i,c in enumerate(clusters):by[c].append(i)
 groups=[np.array(v,int) for v in by.values()];d=np.array([delta[g].sum() for g in groups]);n=np.array([counts[g].sum() for g in groups]);rng=np.random.default_rng(42);boot=[]
 for _ in range(repeats):
  ids=rng.integers(0,len(groups),len(groups));boot.append(d[ids].sum()/n[ids].sum())
 return {'delta':float(delta.sum()/counts.sum()),'ci95_low':float(np.percentile(boot,2.5)),'ci95_high':float(np.percentile(boot,97.5)),'clusters':len(groups),'repeats':repeats,'seed':42}
def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--groups',nargs='+',default=['fft','sst','ha_only']);a=p.parse_args();r=a.project;assert not a.output.exists();a.output.mkdir(parents=True);root=r/'results/next_stage/dev_curves_v1';plan=json.loads((r/'configs/next_stage/curves-v1.json').read_text());rows=[];contrasts=[];sources={};preds={};nll={}
 for group in a.groups:
  assert json.loads((root/(group+'-worker.json')).read_text())['status']=='COMPLETE'
  nodes=plan['groups'][group]
  if group=='fft':nodes=[x for x in nodes if x['relative_output']!='base/step000']
  for node in nodes:
   folder=root/(node.get('reuse') or node['relative_output']);step=0 if node.get('reuse') else node['step'];d=json.loads((folder/'DONE.json').read_text());assert d['status']=='EXPOSED_DIAGNOSTIC_COMPLETE';sources[str(folder/'DONE.json')]=sha(folder/'DONE.json')
   assert sha(folder/'mcq_predictions.jsonl')==d['prediction_sha256'];assert sha(folder/'nll_predictions.jsonl')==d['nll_prediction_sha256'];preds[group,step]={(x['task'],x['language'],x['id']):x for x in map(json.loads,(folder/'mcq_predictions.jsonl').open())};nll[group,step]={x['id']:x for x in map(json.loads,(folder/'nll_predictions.jsonl').open())}
   for key,v in d['mcq'].items():
    task,lang=key.split('|');rows.append({'group':group,'step':step,'task':task,'language':lang,**v})
   rows.append({'group':group,'step':step,'task':'heldout_assistant_nll','language':'ha',**d['heldout_ha_nll']})
  for step in [100,200,300,400]:
   b=preds[group,0];t=preds[group,step];assert b.keys()==t.keys()
   for task in ['reading_exposed','knowledge']:
    for lang in ['en','sw','ha']:
     keys=sorted(k for k in b if k[:2]==(task,lang));clusters=[k[2].rsplit('|',1)[0] if task=='reading_exposed' else k[2] for k in keys]
     for metric in ['correct','correct_answer_4option_log_probability','gold_minus_best_wrong_margin']:
      delta=[float(t[k][metric])-float(b[k][metric]) for k in keys];v=interval(delta,[1]*len(keys),clusters);factor=100 if metric=='correct' else 1
      contrasts.append({'group':group,'reference_step':0,'step':step,'task':task,'language':lang,'metric':metric,'n':len(keys),'unit':'percentage_points' if factor==100 else 'nats','cluster_unit':'passage' if task=='reading_exposed' else 'semantic_question',**{k:(x*factor if k in ['delta','ci95_low','ci95_high'] else x) for k,x in v.items()}})
   b=nll[group,0];t=nll[group,step];assert b.keys()==t.keys();keys=sorted(b);assert all(b[k]['assistant_tokens']==t[k]['assistant_tokens'] for k in keys)
   v=interval([t[k]['nll_sum']-b[k]['nll_sum'] for k in keys],[b[k]['assistant_tokens'] for k in keys],keys);contrasts.append({'group':group,'reference_step':0,'step':step,'task':'heldout_assistant_nll','language':'ha','metric':'token_weighted_nll','n':len(keys),'unit':'nats_per_assistant_token','cluster_unit':'original_heldout_example',**v})
 # Training losses are native logged optimization quantities, not heldout NLL.
 loss=[]
 for group in a.groups:
  path=Path(next(x['model'] for x in plan['groups'][group] if x.get('step')==400))/'trainer_log.jsonl'
  if path.exists():
   sources[str(path)]=sha(path)
   for x in map(json.loads,path.open()):
    if 'loss' in x:loss.append({'group':group,'step':x['current_steps'],'loss':x['loss'],'lr':x.get('lr')})
 result={'status':'COMPLETE_FOR_LISTED_GROUPS','groups':a.groups,'rows':rows,'paired_step_changes':contrasts,'training_loss':loss,'sources_sha256':sources,'protocol_sha256':sha(Path(plan['data'])/'protocol.json'),'test_used_for_selection':False,'bootstrap':'2000 paired resamples; reading whole-passage clusters, knowledge original semantic IDs, NLL original heldout examples and token-weighted numerator/denominator; unadjusted descriptive intervals','limits':['Exposed diagnostic items; no novel-method confirmation','All points retained;Final fixed400','NLL and choice confidence auxiliary, not downstream success criteria','Different initialW1 acrossFFT andSST; do not attribute differences solely to constraint']}
 (a.output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(a.output/'curve_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (a.output/'curve_rows.csv').open('w',newline='') as f:
  keys=['group','step','task','language','n','acc','correct_answer_4option_log_probability','gold_minus_best_wrong_margin','token_weighted_nll','assistant_tokens'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
 lines=['# 全部冻结开发节点','','阅读为已曝光180题，知识为已曝光114题；NLL为独立heldout256/30742监督token，均非正式全量成绩。','','| Group | HA step | HA reading ACC | HA knowledge ACC | HA heldout NLL | SW reading ACC | SW knowledge ACC |','|---|---:|---:|---:|---:|---:|---:|']
 lookup={(x['group'],x['step'],x['task'],x['language']):x for x in rows}
 for group in a.groups:
  for step in [0,100,200,300,400]:
   vals=[]
   for task,lang,metric in [('reading_exposed','ha','acc'),('knowledge','ha','acc'),('heldout_assistant_nll','ha','token_weighted_nll'),('reading_exposed','sw','acc'),('knowledge','sw','acc')]:
    v=lookup[group,step,task,lang][metric];vals.append(f'{v*100:.2f}' if metric=='acc' else f'{v:.4f}')
   lines.append('| '+group+' | '+str(step)+' | '+' | '.join(vals)+' |')
 lines+=['','全部置信区间、正确答案log-probability、margin、位置诊断和训练loss见analysis.json；不挑最高节点替换Final。']
 (a.output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'groups':a.groups,'rows':len(rows),'paired_contrasts':len(contrasts)}))
if __name__=='__main__':main()
