"""Progressive complete-scope score table. Missing results remain empty, never zero."""
from pathlib import Path
import csv,hashlib,json,datetime
R=Path('/path/to/project')
Q=R/'research/function_aware_formal_20261109'
L=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109/evaluation')
TASKS=['target_reading','target_summary','target_knowledge','target_xnli','en_gsm8k','en_ifeval','en_mmlu','en_code','en_reading','en_xnli']
NAMES={'target_reading':'目标Belebele ACC%','target_summary':'目标XL-Sum ROUGE-Lsum','target_knowledge':'目标Global-MMLU ACC%','target_xnli':'目标XNLI ACC%','en_gsm8k':'GSM8K flexible ACC%','en_ifeval':'IFEval prompt strict%','en_mmlu':'EN MMLU ACC%','en_code':'HumanEval+ pass@1%','en_reading':'EN Belebele ACC%','en_xnli':'EN XNLI ACC%'}
LABELS={'Base':'原模型','all':'普通LoRA','full':'均匀表示保持（72点）','kd':'Logit KD','random':'随机加权表示保持（72点）','function':'功能感知候选（72点）','HistoricalFull36':'历史均匀表示保持（36点）'}
def load(p):return json.loads(p.read_text())
def one(root,pattern):
 p=list(root.rglob(pattern));assert len(p)==1,(root,pattern,len(p));return p[0]
def main():
 scores={};sources={};stats={}
 def put(lang,method,task,value,source):
  key=(lang,method,task);value=float(value)
  assert 0<=value<=100,(key,value)
  if key in scores:assert abs(scores[key]-value)<1e-7,(key,scores[key],value)
  scores[key]=value;sources[key]=str(source)
 source=R/'research/representation_benchmarks_20261109/complete_scores.csv'
 mapping={('Belebele target','ACC'):'target_reading',('Belebele EN','ACC'):'en_reading',('XL-Sum target','ROUGE-Lsum'):'target_summary',('Global-MMLU SW','ACC'):'target_knowledge',('GSM8K','flexible-extract'):'en_gsm8k',('IFEval','prompt_level_strict_acc'):'en_ifeval',('XNLI SW','ACC'):'target_xnli',('XNLI EN','ACC'):'en_xnli',('HumanEval+','pass@1'):'en_code'}
 for row in csv.DictReader(source.open()):
  task=mapping.get((row['task'],row['metric']))
  if task is None:continue
  for col,method in [('Base','Base'),('all_layer','all'),('full_feature','HistoricalFull36')]:put('sw',method,task,row[col],source)
  if task.startswith('en_'):
   for lang in ['ar','ru']:put(lang,'Base',task,row['Base'],source)
 base_mmlu=R/'results/aya_4b_20261105/analysis-v1/scores.csv'
 for row in csv.DictReader(base_mmlu.open()):
  if row['model']=='base' and row['task']=='MMLU EN' and row['metric']=='ACC':
   assert row['status']=='VERIFIED_COMPLETE'
   for lang in ['sw','ar','ru']:put(lang,'Base','en_mmlu',row['score'],base_mmlu)
 historical_mmlu=Path('/path/to/logs/qwen3-multilingual/fft-representation-20261109/evaluation/p2_full36/mmlu')
 if (historical_mmlu/'DONE.json').exists():
  f=historical_mmlu/'scoring.log';m=load(f);assert m['status']=='validated' and m['total_count']==14042 and len(m['tasks'])==57
  put('sw','HistoricalFull36','en_mmlu',100*m['micro_accuracy'],f)
 historical=R/'research/aya_4b_generalization_20261107/results/scores.csv'
 for row in csv.DictReader(historical.open()):
  task={('Belebele target','ACC'):'target_reading',('XL-Sum target','ROUGE-Lsum'):'target_summary',('XNLI target','ACC'):'target_xnli'}.get((row['task'],row['metric']))
  if task:put(row['language'],'Base',task,row['base'],historical)
 for lang in ['sw','ar','ru']:
  for method in ['Base','all','full','kd','random','function']:
   root=L/(lang+'_'+method)
   for stage in ['reading','summary','gmmlu','mmlu','core','code']:
    folder=root/stage
    if not (folder/'DONE.json').exists():continue
    marker=load(folder/'DONE.json');assert marker['status']=='FIXED_ENDPOINT_STAGE_COMPLETE_PENDING_PAIRED_AUDIT'
    if stage=='reading':
     f=one(folder,'results_*.json');results=load(f)['results']
     pairs=[('target_reading','belebele_swh_Latn_locked' if lang=='sw' else 'aya4bml_belebele_'+lang),('en_reading','belebele_eng_Latn_locked'),('target_xnli','rb_xnli_sw_formal_primary' if lang=='sw' else 'aya4bml_xnli_'+lang),('en_xnli','rb_xnli_en_formal_primary')]
     for task,name in pairs:put(lang,method,task,100*results[name]['acc,none'],f)
    elif stage in ['gmmlu','mmlu']:
     f=folder/'scoring.log';audit=json.loads(f.read_text());assert audit['status']=='validated' and audit['total_count']==14042 and len(audit['tasks'])==57
     value=100*audit['micro_accuracy'];task='target_knowledge' if stage=='gmmlu' else 'en_mmlu';put(lang,method,task,value,f)
     if lang=='sw' and method=='Base' and stage=='mmlu':
      for other in ['ar','ru']:put(other,method,task,value,f)
    elif stage=='core':
     f=one(folder,'results_*.json');res=load(f)['results'];put(lang,method,'en_gsm8k',100*res['sft_v1_gsm8k']['exact_match,flexible-extract'],f);put(lang,method,'en_ifeval',100*res['sft_v1_ifeval']['prompt_level_strict_acc,none'],f)
    elif stage=='summary':
     f=folder/'scores'/('summary.json' if lang=='sw' else 'scores.json');m=load(f);put(lang,method,'target_summary',m['metrics']['ROUGE-Lsum'],f)
    else:
     f=folder/'execution/scores.json';m=load(f);assert m['status']=='COMPLETE' and m['n']==164;put(lang,method,'en_code',100*m['pass_at_1_plus'],f)
 rows=[]
 for lang in ['sw','ar','ru']:
  for method in ['Base','all','full','kd','random','function']+(['HistoricalFull36'] if lang=='sw' else []):
   for task in TASKS:
    key=(lang,method,task);rows.append(dict(language=lang,method=method,task=task,score=scores.get(key,''),status='AVAILABLE_SAME_PROTOCOL' if key in scores else 'NOT_COMPLETE',source=sources.get(key,'')))
 with (Q/'scores_long.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 wide=[]
 for lang in ['sw','ar','ru']:
  for method in ['Base','all','full','kd','random','function']+(['HistoricalFull36'] if lang=='sw' else []):wide.append(dict(language=lang,method=method,**{task:scores.get((lang,method,task),'') for task in TASKS}))
 with (Q/'scores_wide.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(wide[0]));w.writeheader();w.writerows(wide)
 md=['# 三语言正式主结果表（持续更新）','',datetime.datetime.now().astimezone().isoformat(),'', '缺失结果为“—”，不是0。HistoricalFull36为旧SW36点表示保持，只作历史补充；新主对照full使用共同72点。各语言是独立单语适配，不构成顺序CL证据。','', 'Base通用指标是相同模型、相同冻结协议的共享参考；旧SW batch2普通LoRA复用。AR/RU旧batch1 LoRA不混入新主表。当前新方法尚未全部完成，不能作优劣结论。']
 for lang in ['sw','ar','ru']:
  md+=['','## '+lang.upper(),'', '| 方法 | '+' | '.join(NAMES[t] for t in TASKS)+' |','|---|'+'---:|'*len(TASKS)]
  for row in wide:
   if row['language']!=lang:continue
   md.append('| '+LABELS[row['method']]+' | '+' | '.join('—' if row[t]=='' else f'{row[t]:.2f}' for t in TASKS)+' |')
 md+=['','仅在同语言、同任务内解释差值；SW与AR/RU摘要采用各自原冻结分词协议，不比较跨语言ROUGE绝对大小。XNLI选项偏好限制保留。所有训练固定Final400，无正式test选模；当前表尚待逐题配对审计及区间。']
 (Q/'results.md').write_text('\n'.join(md)+'\n')
 print(json.dumps(dict(available_cells=len(scores),new_main_table_cells=180,updated_at=datetime.datetime.now().astimezone().isoformat())))
if __name__=='__main__':main()
