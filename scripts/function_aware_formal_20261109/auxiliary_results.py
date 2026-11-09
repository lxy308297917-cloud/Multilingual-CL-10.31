"""Retain GSM8K strict and IFEval instruction strict for every formal model."""
from pathlib import Path
import csv,json,datetime
import numpy as np
import paired_results as audit
from collect_results import LABELS
Q=audit.Q
TASKS=['GSM8K strict','IFEval instruction strict']
def main():
 vectors={};scores=[];sources=[];missing=[];numeric=Q/'auxiliary_numeric';numeric.mkdir(exist_ok=True)
 for lang in ['sw','ar','ru']:
  for method in ['Base','all','full','kd','random','function']+(['HistoricalFull36'] if lang=='sw' else []):
   for task in TASKS:
    found=audit.locate(lang,method,'en_gsm8k' if task==TASKS[0] else 'en_ifeval')
    if found is None:missing.append((lang,method,task));continue
    _,files=found;assert len(files)==1;path=Path(files[0]['path']);result=audit.load(audit.only(path.parent,'results_*.json'))['results'];v={}
    for row in audit.rows(path):
     if task==TASKS[0]:
      if row['filter']!='strict-match':continue
      numerator=float(row['exact_match']);denominator=1
     else:
      assert row['filter']=='none';hits=row['inst_level_strict_acc'];numerator=sum(bool(x) for x in hits);denominator=len(hits);assert bool(row['prompt_level_strict_acc'])==all(hits)
     key=str(row['doc_id']);assert key not in v;v[key]=dict(identity=audit.signature(row),numerator=numerator,denominator=denominator)
    count=1319 if task==TASKS[0] else 541;denominator=1319 if task==TASKS[0] else 834
    assert len(v)==count and sum(x['denominator'] for x in v.values())==denominator
    score=100*sum(x['numerator'] for x in v.values())/denominator
    target=result['sft_v1_gsm8k']['exact_match,strict-match'] if task==TASKS[0] else result['sft_v1_ifeval']['inst_level_strict_acc,none'];assert abs(score-100*target)<1e-8
    vectors[lang,method,task]=v;scores.append(dict(language=lang,method=method,task=task,score=score,prompts=count,score_denominator=denominator));sources.append(dict(language=lang,method=method,task=task,path=str(path),sha256=audit.sha(path)))
 for lang in ['sw','ar','ru']:
  for ti,task in enumerate(TASKS):
   models=[m for m in ['Base','all','full','kd','random','function','HistoricalFull36'] if (lang,m,task) in vectors]
   if not models:continue
   first=vectors[lang,models[0],task];keys=sorted(first)
   for model in models:
    v=vectors[lang,model,task];assert v.keys()==first.keys() and all(v[k]['identity']==first[k]['identity'] and v[k]['denominator']==first[k]['denominator'] for k in keys)
   data=dict(language=lang,task=task,models=models,item_id_sha256=[audit.digest(k) for k in keys],denominators=[first[k]['denominator'] for k in keys],numerators=[[vectors[lang,m,task][k]['numerator'] for m in models] for k in keys],ordered_input_identity_sha256=audit.digest([(k,first[k]['identity']) for k in keys]),bootstrap_unit='whole prompt',no_raw_text=True)
   (numeric/(lang+'_'+str(ti)+'.json')).write_text(json.dumps(data,separators=(',',':'))+'\n')
 contrasts=[('Base','all'),('Base','full'),('Base','kd'),('Base','random'),('Base','function'),('all','full'),('all','kd'),('all','random'),('all','function'),('full','function'),('random','function'),('kd','function'),('HistoricalFull36','function')];intervals=[]
 for li,lang in enumerate(['sw','ar','ru']):
  for ci,(a,b) in enumerate(contrasts):
   for ti,task in enumerate(TASKS):
    ka,kb=(lang,a,task),(lang,b,task)
    if ka not in vectors or kb not in vectors:continue
    va,vb=vectors[ka],vectors[kb];keys=sorted(va);assert va.keys()==vb.keys() and all(va[k]['identity']==vb[k]['identity'] and va[k]['denominator']==vb[k]['denominator'] for k in keys)
    dif=np.asarray([vb[k]['numerator']-va[k]['numerator'] for k in keys]);den=np.asarray([va[k]['denominator'] for k in keys]);rng=np.random.default_rng(np.random.SeedSequence([42,li,ci,ti]));deltas=[]
    for _ in range(2000):
     idx=rng.integers(len(keys),size=len(keys));deltas.append(100*dif[idx].sum()/den[idx].sum())
    lo,hi=np.quantile(deltas,[.025,.975]);intervals.append(dict(language=lang,reference=a,comparison=b,task=task,delta=float(100*dif.sum()/den.sum()),ci95_low=float(lo),ci95_high=float(hi),score_denominator=int(den.sum()),prompt_clusters=len(keys),resamples=2000))
 for filename,data in [('auxiliary_scores.csv',scores),('auxiliary_paired_changes.csv',intervals)]:
  if data:
   with (Q/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 report=dict(status='INCOMPLETE' if missing else 'ALL_AUXILIARY_CELLS_RAW_AUDITED',verified_cells=len(scores),required_cells=38,missing=missing,paired_contrasts=len(intervals),IFEval_score_denominator=834,IFEval_resampling_clusters=541,bootstrap_resamples=2000,no_mean_prompt_instruction_rate=True,not_training_seed_uncertainty=True)
 (Q/'auxiliary_audit.json').write_text(json.dumps(report,indent=2)+'\n');(Q/'auxiliary_provenance.json').write_text(json.dumps(sources,indent=2)+'\n')
 md=['# 三语言通用能力补充指标','',datetime.datetime.now().astimezone().isoformat(),'', 'GSM8K strict与主表flexible并列保留；IFEval instruction strict以834条指令为分母，区间按541个完整prompt重采样。—不是0；区间是固定模型的样本不确定性，不代表训练种子稳定性。','', '| 语言 | 模型 | GSM8K strict ACC% | IFEval instruction strict% |','|---|---|---:|---:|']
 lookup={(x['language'],x['method'],x['task']):x['score'] for x in scores}
 for lang in ['sw','ar','ru']:
  for method in ['Base','all','full','kd','random','function']+(['HistoricalFull36'] if lang=='sw' else []):md.append('| '+lang.upper()+' | '+LABELS[method]+' | '+' | '.join('—' if (lang,method,t) not in lookup else f'{lookup[lang,method,t]:.2f}' for t in TASKS)+' |')
 (Q/'auxiliary_results.md').write_text('\n'.join(md)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='missing'}))
if __name__=='__main__':main()
