"""Progressive FFT comparison. Only completed scoring is included; no missing-as-zero."""
from pathlib import Path
import csv,json,datetime
R=Path('/path/to/project')
Q=R/'research/fft_representation_20261109'
L=Path('/path/to/logs/qwen3-multilingual/fft-representation-20261109')
PROFILES=['Base','historical_fft','matched_fft','fft_full36','p2_all','p2_full36','p2_ce','historical_ssu']
TASKS=['SW Belebele','SW XL-Sum','SW Global-MMLU','SW XNLI','EN Belebele','EN XNLI','EN MMLU','GSM8K flexible','IFEval prompt strict','HumanEval+','EN→SW chrF++','SW→EN chrF++']
LABELS={'Base':'原模型','historical_fft':'历史普通FFT','matched_fft':'新匹配普通FFT','fft_full36':'新FFT＋36点表示保持','p2_all':'P2普通LoRA','p2_full36':'P2 LoRA＋36点表示保持','p2_ce':'P2 LoRA＋CE回放','historical_ssu':'历史SSU'}
def load(p):return json.loads(p.read_text())
def one(root,pattern):
 paths=list(root.rglob(pattern));assert len(paths)==1,(root,pattern,len(paths));return paths[0]
def main():
 scores={};sources={}
 def put(profile,task,value,source):
  value=float(value);assert 0<=value<=100
  k=(profile,task)
  if k in scores:assert abs(scores[k]-value)<1e-7,(k,scores[k],value)
  scores[k]=value;sources[k]=str(source)
 hist=R/'results/aya_4b_20261105/analysis-v1/scores.csv'
 hm={('Belebele SW','ACC'):'SW Belebele',('Belebele EN','ACC'):'EN Belebele',('Global-MMLU SW','ACC'):'SW Global-MMLU',('MMLU EN','ACC'):'EN MMLU',('Summary SW','ROUGE-Lsum'):'SW XL-Sum',('GSM8K flexible-extract','ACC'):'GSM8K flexible',('IFEval','prompt_level_strict_acc'):'IFEval prompt strict',('Translation EN->SW','chrF++'):'EN→SW chrF++',('Translation SW->EN','chrF++'):'SW→EN chrF++'}
 profiles={'base':'Base','fft':'historical_fft','ssu':'historical_ssu'}
 for row in csv.DictReader(hist.open()):
  task=hm.get((row['task'],row['metric']));profile=profiles.get(row['model'])
  if task and profile:
   assert row['status']=='VERIFIED_COMPLETE';put(profile,task,row['score'],hist)
 xn=R/'results/aya_4b_20261105/supplemental-analysis-v1/scores.json'
 for row in load(xn):
  if row['model'] in profiles and row['task'] in ['xnli_sw','xnli_en']:
   assert row['status']=='VERIFIED_COMPLETE';put(profiles[row['model']],'SW XNLI' if row['task']=='xnli_sw' else 'EN XNLI',row['score'],xn)
 rb=R/'research/representation_benchmarks_20261109/complete_scores.csv'
 rm={('Belebele target','ACC'):'SW Belebele',('Belebele EN','ACC'):'EN Belebele',('XL-Sum target','ROUGE-Lsum'):'SW XL-Sum',('Global-MMLU SW','ACC'):'SW Global-MMLU',('GSM8K','flexible-extract'):'GSM8K flexible',('IFEval','prompt_level_strict_acc'):'IFEval prompt strict',('XNLI SW','ACC'):'SW XNLI',('XNLI EN','ACC'):'EN XNLI',('HumanEval+','pass@1'):'HumanEval+'}
 for row in csv.DictReader(rb.open()):
  task=rm.get((row['task'],row['metric']))
  if task:
   for col,profile in [('Base','Base'),('all_layer','p2_all'),('full_feature','p2_full36'),('ce_replay','p2_ce')]:put(profile,task,row[col],rb)
 # Main-track P2 all-layer EN MMLU is already queued there; do not duplicate inference.
 fa=R/'research/function_aware_formal_20261109/scores_long.csv'
 for row in csv.DictReader(fa.open()):
  if row['language']=='sw' and row['method']=='all' and row['task']=='en_mmlu' and row['score']!='':put('p2_all','EN MMLU',row['score'],row['source'])
 for profile in PROFILES:
  root=L/'evaluation'/profile
  for stage in ['reading','summary','gmmlu','mmlu','core','code','mt_en_sw','mt_sw_en']:
   folder=root/stage
   if not (folder/'DONE.json').exists():continue
   assert load(folder/'DONE.json')['status']=='FIXED_ENDPOINT_STAGE_COMPLETE_PENDING_PAIRED_AUDIT'
   if stage=='reading':
    source=one(folder,'results_*.json');res=load(source)['results']
    for task,name in [('SW Belebele','belebele_swh_Latn_locked'),('EN Belebele','belebele_eng_Latn_locked'),('SW XNLI','rb_xnli_sw_formal_primary'),('EN XNLI','rb_xnli_en_formal_primary')]:put(profile,task,100*res[name]['acc,none'],source)
   elif stage in ['gmmlu','mmlu']:
    source=folder/'scoring.log';m=load(source);assert m['status']=='validated' and m['total_count']==14042 and len(m['tasks'])==57
    put(profile,'SW Global-MMLU' if stage=='gmmlu' else 'EN MMLU',100*m['micro_accuracy'],source)
   elif stage=='core':
    source=one(folder,'results_*.json');res=load(source)['results'];put(profile,'GSM8K flexible',100*res['sft_v1_gsm8k']['exact_match,flexible-extract'],source);put(profile,'IFEval prompt strict',100*res['sft_v1_ifeval']['prompt_level_strict_acc,none'],source)
   elif stage=='code':
    source=folder/'execution/scores.json';m=load(source);assert m['status']=='COMPLETE' and m['n']==164;put(profile,'HumanEval+',100*m['pass_at_1_plus'],source)
   else:
    source=folder/'scores/summary.json';m=load(source);task={'summary':'SW XL-Sum','mt_en_sw':'EN→SW chrF++','mt_sw_en':'SW→EN chrF++'}[stage];put(profile,task,m['metrics']['ROUGE-Lsum' if stage=='summary' else 'chrF++'],source)
 rows=[dict(profile=p,task=t,score=scores.get((p,t),''),status='COMPLETE_SCORE_PENDING_CROSS_FAMILY_PAIRED_AUDIT' if (p,t) in scores else 'NOT_COMPLETE',source=sources.get((p,t),'')) for p in PROFILES for t in TASKS]
 with (Q/'scores.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 md=['# FFT补充实验结果（持续更新）','',datetime.datetime.now().astimezone().isoformat(),'', '历史已完成分数复用；新增结果只在正式评分完成后填写。缺失为—，不是0。当前没有新FFT方法成效结论。','', '| 模型 | '+' | '.join(TASKS)+' |','|---|'+'---:|'*len(TASKS)]
 for p in PROFILES:md.append('| '+LABELS[p]+' | '+' | '.join('—' if (p,t) not in scores else f'{scores[p,t]:.2f}' for t in TASKS)+' |')
 md+=['','摘要为ROUGE-Lsum，翻译为语料级chrF++，其余为百分比。新FFT使用同实现普通FFT配对；历史FFT另列。LoRA与FFT学习率/参数量不同，不能把绝对差异单独归因于参数化。尚待原始逐题配对与成本审计。']
 (Q/'results.md').write_text('\n'.join(md)+'\n')
 states={}
 for mode in ['none','full36']:
  f=L/mode/'state.json';states[mode]=load(f) if f.exists() else dict(status='NOT_STARTED')
 schedule=load(L/'evaluation-schedule.json') if (L/'evaluation-schedule.json').exists() else {}
 result=dict(updated_at=datetime.datetime.now().astimezone().isoformat(),training=states,available_score_cells=len(scores),total_score_cells=len(PROFILES)*len(TASKS),evaluation=schedule,missing_score_is_not_zero=True)
 (Q/'progress.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(available=len(scores),total=len(PROFILES)*len(TASKS),training={k:v['status'] for k,v in states.items()})))
if __name__=='__main__':main()
