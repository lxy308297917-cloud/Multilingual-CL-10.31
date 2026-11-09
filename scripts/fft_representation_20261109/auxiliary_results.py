"""Required strict GSM8K, instruction-level IFEval and corpus BLEU supplements."""
from pathlib import Path
import csv,json,datetime
import numpy as np
from collect_results import R,Q,L as LOG,PROFILES,LABELS
from paired_results import audit,OLD,P2,translation
AUX=['GSM8K strict','IFEval instruction strict','EN→SW BLEU','SW→EN BLEU']
def main():
 from sacrebleu.metrics import BLEU
 import importlib.metadata
 assert importlib.metadata.version('sacrebleu')=='2.6.0'
 scores={};sources={}
 def put(profile,task,value,source):
  key=(profile,task);value=float(value);assert 0<=value<=100
  if key in scores:assert abs(scores[key]-value)<1e-8
  scores[key]=value;sources[key]=str(source)
 path=R/'results/aya_4b_20261105/analysis-v1/scores.csv'
 mapping={('GSM8K strict-match','ACC'):AUX[0],('IFEval','inst_level_strict_acc'):AUX[1],('Translation EN->SW','BLEU'):AUX[2],('Translation SW->EN','BLEU'):AUX[3]}
 for row in csv.DictReader(path.open()):
  profile={'base':'Base','fft':'historical_fft','ssu':'historical_ssu'}.get(row['model']);task=mapping.get((row['task'],row['metric']))
  if profile and task:assert row['status']=='VERIFIED_COMPLETE';put(profile,task,row['score'],path)
 path=R/'research/representation_benchmarks_20261109/complete_scores.csv'
 for row in csv.DictReader(path.open()):
  task={('GSM8K','strict-match'):AUX[0],('IFEval','inst_level_strict_acc'):AUX[1]}.get((row['task'],row['metric']))
  if task:
   for col,profile in [('Base','Base'),('all_layer','p2_all'),('full_feature','p2_full36'),('ce_replay','p2_ce')]:put(profile,task,row[col],path)
 for profile in PROFILES:
  root=LOG/'evaluation'/profile
  if (root/'core/DONE.json').exists():
   path=audit.only(root/'core','results_*.json');r=audit.load(path)['results'];put(profile,AUX[0],100*r['sft_v1_gsm8k']['exact_match,strict-match'],path);put(profile,AUX[1],100*r['sft_v1_ifeval']['inst_level_strict_acc,none'],path)
  for stage,task in [('mt_en_sw',AUX[2]),('mt_sw_en',AUX[3])]:
   if (root/stage/'DONE.json').exists():
    path=root/stage/'scores/summary.json';put(profile,task,audit.load(path)['metrics']['BLEU'],path)
 vectors={};stats={};proof=[];numeric=Q/'auxiliary_numeric';numeric.mkdir(exist_ok=True)
 for (profile,task),score in scores.items():
  if 'BLEU' in task:
   xs,files=translation(profile,task);scorer=BLEU(lowercase=False,tokenize='13a',smooth_method='exp',smooth_value=None,effective_order=False)
   array=np.asarray(scorer._extract_corpus_statistics([x['prediction'] for x in xs],[[x['reference'] for x in xs]]),dtype=np.int64)
   identity=[audit.digest([x['id'],x['source'],x['reference'],x['prompt_token_sha256']]) for x in xs]
   value=scorer._compute_score_from_stats(array.sum(0).tolist()).score;stats[profile,task]=(array,identity,scorer)
   data=dict(profile=profile,task=task,statistics=array.tolist(),item_id_sha256=[audit.digest(x['id']) for x in xs],ordered_input_identity_sha256=audit.digest(identity),sacrebleu='2.6.0',score=value,no_raw_text=True)
  else:
   root=LOG/'evaluation'/profile/'core'
   if not (root/'DONE.json').exists():
    if profile in ['Base','historical_fft','historical_ssu']:root=OLD/({'Base':'base','historical_fft':'fft','historical_ssu':'ssu'}[profile]+'-generation-v2/core/results')
    else:root=P2/{'p2_all':'all_layer','p2_full36':'full_feature','p2_ce':'ce_replay'}[profile]/'core/results'
   name='sft_v1_gsm8k' if task==AUX[0] else 'sft_v1_ifeval';path=audit.only(root,'samples_'+name+'_*.jsonl');vector={}
   for row in audit.rows(path):
    if task==AUX[0]:
     if row['filter']!='strict-match':continue
     numerator=float(row['exact_match']);denominator=1
    else:
     assert row['filter']=='none';hits=row['inst_level_strict_acc'];numerator=sum(bool(x) for x in hits);denominator=len(hits)
     assert bool(row['prompt_level_strict_acc'])==all(hits)
    key=str(row['doc_id']);assert key not in vector;vector[key]=dict(numerator=numerator,denominator=denominator,identity=audit.signature(row))
   assert len(vector)==(1319 if task==AUX[0] else 541)
   assert sum(x['denominator'] for x in vector.values())==(1319 if task==AUX[0] else 834)
   value=100*sum(x['numerator'] for x in vector.values())/sum(x['denominator'] for x in vector.values());vectors[profile,task]=vector;keys=sorted(vector)
   data=dict(profile=profile,task=task,item_id_sha256=[audit.digest(str(k)) for k in keys],rows=[[vector[k]['numerator'],vector[k]['denominator']] for k in keys],ordered_input_identity_sha256=audit.digest([(k,vector[k]['identity']) for k in keys]),row_unit='source prompt',score=value,no_raw_text=True)
   files=[dict(path=str(path),sha256=audit.sha(path))]
  assert abs(value-score)<1e-8,(profile,task,value,score)
  proof.append(dict(profile=profile,task=task,files=files));(numeric/(profile+'_'+str(AUX.index(task))+'.json')).write_text(json.dumps(data,separators=(',',':'))+'\n')
 contrasts=[('Base','historical_fft'),('Base','matched_fft'),('Base','fft_full36'),('historical_fft','matched_fft'),('matched_fft','fft_full36'),('p2_all','p2_full36'),('p2_full36','fft_full36'),('p2_ce','fft_full36'),('historical_ssu','fft_full36')];intervals=[]
 for ci,(a,b) in enumerate(contrasts):
  for ti,task in enumerate(AUX):
   ka,kb=(a,task),(b,task)
   if ka not in scores or kb not in scores:continue
   rng=np.random.default_rng(np.random.SeedSequence([42,ci,ti]));deltas=[]
   if ka in vectors:
    va,vb=vectors[ka],vectors[kb];assert va.keys()==vb.keys();keys=sorted(va)
    assert all(va[k]['identity']==vb[k]['identity'] and va[k]['denominator']==vb[k]['denominator'] for k in keys)
    dif=np.array([vb[k]['numerator']-va[k]['numerator'] for k in keys]);den=np.array([va[k]['denominator'] for k in keys]);n=len(keys)
    for _ in range(2000):
     idx=rng.integers(n,size=n);deltas.append(100*dif[idx].sum()/den[idx].sum())
    units=int(den.sum());clusters=n
   else:
    sa,ia,scorer=stats[ka];sb,ib,_=stats[kb];assert ia==ib;units=clusters=1012
    for _ in range(2000):
     idx=rng.integers(1012,size=1012);deltas.append(scorer._compute_score_from_stats(sb[idx].sum(0).tolist()).score-scorer._compute_score_from_stats(sa[idx].sum(0).tolist()).score)
   lo,hi=np.quantile(deltas,[.025,.975]);intervals.append(dict(reference=a,comparison=b,task=task,delta=scores[kb]-scores[ka],ci95_low=float(lo),ci95_high=float(hi),score_denominator=units,resampling_clusters=clusters,resamples=2000))
 rows=[dict(profile=p,task=t,score=scores.get((p,t),''),status='RAW_AUDITED' if (p,t) in scores else 'NOT_COMPLETE',source=sources.get((p,t),'')) for p in PROFILES for t in AUX]
 for name,data in [('auxiliary_scores.csv',rows),('auxiliary_paired_changes.csv',intervals)]:
  if data:
   with (Q/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 md=['# 必需的补充指标','',datetime.datetime.now().astimezone().isoformat(),'', 'GSM8K strict与flexible同时保留；IFEval指令级以834条指令为分母，区间按541个完整提示重采样；翻译BLEU使用语料充分统计量。缺失不是0。','', '| 模型 | '+' | '.join(AUX)+' |','|---|'+'---:|'*4]
 for p in PROFILES:md.append('| '+LABELS[p]+' | '+' | '.join('—' if (p,t) not in scores else f'{scores[p,t]:.2f}' for t in AUX)+' |')
 (Q/'auxiliary_results.md').write_text('\n'.join(md)+'\n');(Q/'auxiliary_provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
 report=dict(status='ALL_AUXILIARY_CELLS_RAW_AUDITED' if len(scores)==32 else 'INCOMPLETE',verified_cells=len(scores),required_cells=32,paired_contrasts=len(intervals),IFEval_instruction_denominator=834,IFEval_prompt_clusters=541,no_mean_sentence_BLEU=True,no_mean_prompt_instruction_rate=True,bootstrap_resamples=2000,not_training_seed_uncertainty=True)
 (Q/'auxiliary_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
