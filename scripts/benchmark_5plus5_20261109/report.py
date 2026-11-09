"""CPU-only raw score checks and five-target/five-general result projection."""
from pathlib import Path
import collections,csv,datetime,hashlib,json,itertools
import numpy as np
R=Path('/path/to/project');Q=R/'research/benchmark_5plus5_20261109';C=R/'configs/benchmark_5plus5_20261109';L=Path('/path/to/logs/qwen3-multilingual/benchmark-5plus5-20261109');D=Path('/path/to/storage/datasets/benchmark-5plus5-20261109/v1')
LABELS=['science/technology','travel','politics','sports','health','entertainment','geography']
TARGET=['Belebele','XNLI','XL-Sum','Global-MMLU','SIB-200'];GENERAL=['GSM8K','IFEval','HumanEval+','EN MMLU','EN Belebele']
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.open() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def only(p,pattern):
 xs=list(p.rglob(pattern));assert len(xs)==1,(p,pattern,len(xs));return xs[0]
def csvrows(p):return list(csv.DictReader(p.open()))
def writecsv(name,data):
 if data:
  with (Q/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def main():
 models=load(C/'models.json');values={};provenance=[];vectors={};sib_reports={};subjects=[];cost=[]
 def put(model,task,value,source):
  if value=='':return
  values[model,task]=float(value);provenance.append(dict(model=model,task=task,source=str(source),source_sha256=sha(source)))
 # Reuse frozen endpoint tables, distinguishing historical adapters from new aligned runs.
 p=R/'research/function_aware_formal_20261109/scores_long.csv';mapping=dict(target_reading='Belebele',target_summary='XL-Sum',target_knowledge='Global-MMLU',target_xnli='XNLI',en_gsm8k='GSM8K',en_ifeval='IFEval',en_code='HumanEval+',en_mmlu='EN MMLU',en_reading='EN Belebele')
 for row in csvrows(p):
  name=row['language']+'_'+row['method']
  if name in models and row['task'] in mapping:put(name,mapping[row['task']],row['score'],p)
 p=R/'research/fft_representation_20261109/scores.csv';mapping={'SW Belebele':'Belebele','SW XNLI':'XNLI','SW XL-Sum':'XL-Sum','SW Global-MMLU':'Global-MMLU','EN Belebele':'EN Belebele','EN MMLU':'EN MMLU','GSM8K flexible':'GSM8K','IFEval prompt strict':'IFEval','HumanEval+':'HumanEval+'}
 for row in csvrows(p):
  if row['profile'] in models and row['task'] in mapping:put(row['profile'],mapping[row['task']],row['score'],p)
 p=R/'research/updates_20261108/aya-4b-generalization-20261107/three-language-scores.csv';mapping={('Belebele target','ACC'):'Belebele',('XNLI target','ACC'):'XNLI',('XL-Sum target','ROUGE-Lsum'):'XL-Sum',('Belebele EN','ACC'):'EN Belebele',('GSM8K','flexible-extract'):'GSM8K',('IFEval','prompt_level_strict_acc'):'IFEval'}
 seen_legacy=set()
 for row in csvrows(p):
  key=(row['task'],row['metric'])
  if key not in mapping:continue
  destination=(row['language']+'_historical_lora',mapping[key]);assert destination not in seen_legacy,('Duplicate historical primary metric',destination);seen_legacy.add(destination)
  if key==('IFEval','prompt_level_strict_acc'):assert int(row['n'])==541
  if key==('GSM8K','flexible-extract'):assert int(row['n'])==1319
  put(*destination,row['final'],p)
 assert len(seen_legacy)==18
 p=R/'research/updates_20261108/sw_original_all_scores.csv';mapping={'Global-MMLU SW':'Global-MMLU','MMLU EN':'EN MMLU'}
 for row in csvrows(p):
  if row['model']=='lora' and row['task'] in mapping:put('sw_historical_lora',mapping[row['task']],row['score'],p)
 import sys
 sys.path.insert(0,str(R/'scripts/function_aware_formal_20261109'));import paired_results as existing_audit
 for name,m in models.items():
  for stage,task in [('mmlu','EN MMLU'),('code','HumanEval+')]:
   folder=L/name/stage
   if not (folder/'DONE.json').exists():continue
   vec,files=existing_audit.mc(folder,'mmlu_*') if stage=='mmlu' else existing_audit.code(folder);put(name,task,np.mean([x['value'] for x in vec.values()]),folder/'DONE.json')
  for task in ['sib200','gmmlu']:
   folder=L/name/task
   if task=='gmmlu' and name in ['ar_Base','ru_Base']:folder=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109/evaluation')/name/'gmmlu'
   if not (folder/'DONE.json').exists():continue
   if task=='gmmlu':
    files=sorted(folder.rglob('samples_global_mmlu_full_'+m['language']+'_*.jsonl'));assert len(files)==57
    allhits=[];v={}
    for file in files:
     xs=rows(file);hits=[]
     for x in xs:
      scores=np.array([r[0] if isinstance(r,(list,tuple)) else r for r in x['filtered_resps']],dtype=float);gold='ABCD'.index(x['target']) if x['target'] in ['A','B','C','D'] else int(x['target']);hit=int(int(np.argmax(scores))==gold);assert hit==x['acc'];key=file.name.split('_20')[0]+':'+str(x['doc_id']);assert key not in v;v[key]=dict(identity=[x['doc_hash'],x['prompt_hash'],x['target_hash']],value=hit);hits.append(hit)
     subjects.append(dict(model=name,language=m['language'],subject=file.name,samples=len(xs),correct=sum(hits),acc=100*np.mean(hits)));allhits.extend(hits)
    assert len(allhits)==14042;put(name,'Global-MMLU',100*np.mean(allhits),folder/'DONE.json');vectors[name,'Global-MMLU']=v
    (Q/'numeric').mkdir(exist_ok=True);keys=sorted(v);(Q/'numeric'/(name+'_global_mmlu.json')).write_text(json.dumps(dict(model=name,language=m['language'],task='Global-MMLU',samples=14042,item_ids=keys,input_identity_sha256=[hashlib.sha256(json.dumps(v[k]['identity']).encode()).hexdigest() for k in keys],values=[100*v[k]['value'] for k in keys]),separators=(',',':'))+'\n')
   else:
    file=only(folder,'samples_mcl5_sib200_'+m['language']+'_*.jsonl');xs=rows(file);assert len(xs)==204;conf=np.zeros((7,7),dtype=int);rawhits=[];v={};numeric=[]
    for x in xs:
     scores=np.array([r[0] if isinstance(r,(list,tuple)) else r for r in x['filtered_resps']],dtype=float);assert scores.shape==(7,) and np.isfinite(scores).all();gold=int(x['doc']['label']);pred=int(np.argmax(scores/np.array([len(t) for t in LABELS])));raw=int(np.argmax(scores));hit=int(pred==gold);assert hit==x['acc_norm'];assert int(raw==gold)==x['acc'];conf[gold,pred]+=1;rawhits.append(raw==gold);key=str(x['doc']['index_id']);assert key not in v;v[key]=dict(identity=[x['doc_hash'],x['prompt_hash'],x['target_hash']],value=hit,raw_value=int(raw==gold));numeric.append(dict(id=key,gold=gold,pred=pred,raw_pred=raw,option_loglikelihoods=scores.tolist(),input_identity_sha256=hashlib.sha256(json.dumps(v[key]['identity']).encode()).hexdigest()))
    aggregate=load(only(folder,'results_*.json'))['results']['mcl5_sib200_'+m['language']];score=100*conf.trace()/204;assert abs(score/100-aggregate['acc_norm,none'])<1e-10 and abs(np.mean(rawhits)-aggregate['acc,none'])<1e-10
    support=conf.sum(1);predcounts=conf.sum(0);f1=np.divide(2*np.diag(conf),support+predcounts,out=np.zeros(7),where=(support+predcounts)>0);sib_reports[name]=dict(samples=204,primary_acc_norm=score,raw_acc=100*np.mean(rawhits),macro_f1=100*f1.mean(),labels=LABELS,class_counts=support.tolist(),prediction_counts=predcounts.tolist(),per_class_recall=np.divide(np.diag(conf)*100,support).tolist(),majority_baseline=100*support.max()/204,confusion_matrix_gold_rows_pred_columns=conf.tolist(),raw_sample_file_sha256=sha(file));vectors[name,'SIB-200']=v;vectors[name,'SIB-200 raw']={k:{**x,'value':x['raw_value']} for k,x in v.items()};put(name,'SIB-200',score,file);(Q/'numeric').mkdir(exist_ok=True);(Q/'numeric'/(name+'_sib200.json')).write_text(json.dumps(numeric,separators=(',',':'))+'\n')
   command=folder/'inference.log.command.json'
   if command.exists():
    start=datetime.datetime.fromisoformat(load(command)['started_at']).timestamp();cost.append(dict(model=name,task=task,postlaunch_wall_seconds_including_model_load_cpu_audit=max(0,(folder/'DONE.json').stat().st_mtime-start),inference_log_bytes=(folder/'inference.log').stat().st_size))
 intervals=[]
 for base,name in itertools.combinations(models,2):
  if models[base]['language']!=models[name]['language']:continue
  for task in ['SIB-200','SIB-200 raw','Global-MMLU']:
   if name==base or (name,task) not in vectors or (base,task) not in vectors:continue
   a,b=vectors[base,task],vectors[name,task];assert a.keys()==b.keys() and all(a[k]['identity']==b[k]['identity'] for k in a),'Changed benchmark inputs';keys=sorted(a);dif=np.array([b[k]['value']-a[k]['value'] for k in keys]);rng=np.random.default_rng(42);draws=[100*np.mean(dif[rng.integers(len(keys),size=len(keys))]) for _ in range(2000)];lo,hi=np.quantile(draws,[.025,.975]);intervals.append(dict(reference=base,model=name,task=task,delta=100*dif.mean(),ci95_low=lo,ci95_high=hi,samples=len(keys),resamples=2000))
 summary=[];deltas=[]
 for name,m in models.items():
  row=dict(language=m['language'],model=name,role=m['role'],execution_status='USER_DEFERRED' if name in load(C/'deferred_models.json') else 'AVAILABLE_OR_RUNNING');row.update({t:values.get((name,t),'') for t in TARGET+GENERAL});accs=[row[t] for t in ['Belebele','XNLI','Global-MMLU','SIB-200']];row['TargetACC4']=sum(accs)/4 if all(x!='' for x in accs) else '';gs=[row[t] for t in GENERAL];row['GeneralAvg5']=sum(gs)/5 if all(x!='' for x in gs) else '';summary.append(row)
  for t in TARGET+GENERAL:
   if (name,t) in values and (m['language']+'_Base',t) in values:deltas.append(dict(language=m['language'],model=name,task=t,score=values[name,t],base=values[m['language']+'_Base',t],delta=values[name,t]-values[m['language']+'_Base',t]))
 macros=[];lookup={r['model']:r for r in summary}
 for method in ['Base','historical_lora','all','full','kd','random','function']:
  names=[l+'_'+method for l in ['sw','ar','ru']];group=[lookup[n] for n in names];ready=all(r['TargetACC4']!='' for r in group);bases=[lookup[l+'_Base'] for l in ['sw','ar','ru']];base_ready=all(r['TargetACC4']!='' for r in bases)
  macros.append(dict(method=method,languages='sw/ar/ru',benchmarks='Belebele/XNLI/Global-MMLU/SIB-200',multilingual_TargetACC4=sum(r['TargetACC4'] for r in group)/3 if ready else '',mean_relative_to_language_Base=sum(r['TargetACC4']-b['TargetACC4'] for r,b in zip(group,bases))/3 if ready and base_ready else '',status='COMPLETE' if ready else 'NOT_READY'))
 writecsv('cross_language_macro.csv',macros)
 writecsv('scores_5plus5.csv',summary);writecsv('base_deltas.csv',deltas);writecsv('paired_changes.csv',intervals);writecsv('gmmlu_subjects.csv',subjects);(Q/'sib_diagnostics.json').write_text(json.dumps(sib_reports,indent=2)+'\n');(Q/'score_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');(Q/'evaluation_costs.json').write_text(json.dumps(dict(scope='Postlaunch wall time, not CUDA kernel time; existing reused task cost not newly charged',records=cost),indent=2)+'\n')
 md=['# 5+5持续结果表','',datetime.datetime.now().astimezone().isoformat(),'','—表示未完成，不是0。历史micro1与新micro2严格分行。SIB主指标为预先固定的Harness acc_norm（标签字符长度归一化）；原始ACC、Macro-F1和混淆矩阵另外完整保留。','', '| 语言 | 模型 | Belebele | XNLI | XL-Sum | Global-MMLU | SIB-200 | TargetACC4 |','|---|---|---:|---:|---:|---:|---:|---:|']
 for row in summary:md.append('| '+row['language']+' | '+row['model']+' | '+' | '.join('—' if row[t]=='' else f'{row[t]:.2f}' for t in TARGET+['TargetACC4'])+' |')
 md+=['','## 通用能力五项','','| 语言 | 模型 | GSM8K | IFEval | HumanEval+ | EN MMLU | EN Belebele | GeneralAvg5 |','|---|---|---:|---:|---:|---:|---:|---:|']
 for row in summary:md.append('| '+row['language']+' | '+row['model']+' | '+' | '.join('—' if row[t]=='' else f'{row[t]:.2f}' for t in GENERAL+['GeneralAvg5'])+' |')
 md+=['','TargetACC4只平均四项ACC；XL-Sum独立报告。GeneralAvg5是异质能力的描述性等权平均，不替代逐项结果。缺任何一项则不算平均。SIB与Belebele存在已记录的文本包含重叠，不能视为独立文本证据。历史通用能力缺项未冒充已完成；既有正式训练和通用测评继续运行。正式分数不用于修改算法、提示、归一化或checkpoint。','']
 (Q/'results.md').write_text('\n'.join(md))
 (Q/'score_audit.json').write_text(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),sib_completed=len(sib_reports),sib_required_available=27-len(load(C/'deferred_models.json')),user_deferred_models=load(C/'deferred_models.json'),samples_per_sib_model=204,primary_length_normalized_LL_argmax_recomputed=True,raw_secondary_LL_argmax_recomputed=True,raw_aggregate_matches_every_item=True,paired_input_identity_verified=True,paired_intervals=len(intervals),resamples=2000,normalization_frozen_before_test=True,not_chosen_max_primary_secondary=True,sib_manifest_sha256=sha(D/'manifest.json'),status='PARTIAL_RAW_AUDIT' if len(sib_reports)!=27-len(load(C/'deferred_models.json')) else 'ALL_AVAILABLE_SIB_RAW_AUDITED_OTHER_TASKS_CHECK_SEPARATELY'),indent=2)+'\n')
 notes=['# SIB标签偏好与错误分析','','主指标固定为字符长度归一化ACC，原始ACC只作敏感性诊断，不选较高分。Macro-F1是七类等权平均；下表描述观测分布，不直接推断语义能力。','','| 模型 | 归一化ACC | 原始ACC | Macro-F1 | 多数类基线 | 预测最多类别 | 该类预测数/204 |','|---|---:|---:|---:|---:|---|---:|']
 for name,d in sib_reports.items():
  k=int(np.argmax(d['prediction_counts']));notes.append(f"| {name} | {d['primary_acc_norm']:.2f} | {d['raw_acc']:.2f} | {d['macro_f1']:.2f} | {d['majority_baseline']:.2f} | {LABELS[k]} | {d['prediction_counts'][k]} |")
 notes+=['','完整逐类召回和混淆矩阵见sib_diagnostics.json；numeric目录保留每题官方ID、金标签、两种预测及七个实际候选分数，可按固定官方源重建具体错误案例。204题的小样本差异需结合paired_changes.csv区间；区间只针对当前固定模型，不代表训练种子稳定性。SIB与Belebele重叠规模见sib_overlap_audit.json，不能将其作为独立文本证据叠加强化结论。','']
 (Q/'sib_error_analysis.md').write_text('\n'.join(notes));print(json.dumps(dict(models=len(models),sib_completed=len(sib_reports),global_mmlu_full_rows=sum(x['task']=='Global-MMLU' for x in provenance if 'DONE.json' in x['source']),paired_intervals=len(intervals),not_all_complete=len(sib_reports)!=len(models))))
if __name__=='__main__':main()
