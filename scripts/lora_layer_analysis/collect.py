"""Full frozen ablation results, paired item/cluster intervals; never selects training layers."""
import os,json,csv,hashlib,statistics
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];L=Path(os.environ['ABLATION_LOG_ROOT']);SW=Path(os.environ['REFERENCE_SW_LOG_ROOT']);G=Path(os.environ['REFERENCE_GENERALIZATION_LOG_ROOT']);O=R/'research/lora_layer_analysis_20261108';C=O
def load(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.open()]
def one(root,pattern):
 xs=list(root.rglob(pattern));assert len(xs)==1,(root,pattern,len(xs));return xs[0]
def samples(root,task):return lines(one(root,'samples_'+task+'_*.jsonl'))
def pair(a,b):
 a={(x['doc_id'],x['filter']):x for x in a};b={(x['doc_id'],x['filter']):x for x in b};assert a.keys()==b.keys()
 for k in a:
  for field in ['doc_hash','prompt_hash','target_hash']:assert a[k][field]==b[k][field],(k,field)
 return [a[k] for k in sorted(a)],[b[k] for k in sorted(a)]
def main():
 plan=load(C/'plan.json');rows=[];checks=[];pending=[];rng=np.random.default_rng(42)
 baselines={(x['language'],x['task'],x['metric']):x for x in csv.DictReader((O/'reference_scores.csv').open())}
 baselines[('sw','Global-MMLU SW','ACC')]={'base':33.37843612021079,'final':33.335707164221624}
 def record(lang,variant,task,metric,a,b,counts=None):
  a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);assert a.shape==b.shape;w=np.ones(len(a)) if counts is None else np.asarray(counts);den=w.sum();av=100*a.sum()/den;bv=100*b.sum()/den;boot=[]
  for _ in range(2000):
   idx=rng.integers(0,len(a),len(a));boot.append(100*(b[idx]-a[idx]).sum()/w[idx].sum())
  lo,hi=np.percentile(boot,[2.5,97.5]);reference=baselines[(lang,task,metric)];assert abs(av-float(reference['final']))<1e-7,(lang,task,metric,av,reference);rows.append(dict(base=float(reference['base']),delta_vs_base=bv-float(reference['base']),language=lang,variant=variant,task=task,metric=metric,full_lora=av,ablated=bv,delta_vs_full=bv-av,ci95_low=float(lo),ci95_high=float(hi),samples_or_instructions=int(den),bootstrap_unit='prompt_cluster' if counts is not None else 'item',bootstrap_resamples=2000))
 for lang in plan['languages']:
  fullhf=SW/'lora-harness-v1/belebele/results' if lang=='sw' else G/(lang+'-lora-pipeline-v1/hf/results')
  fullcore=SW/'lora-generation-v2/core/results' if lang=='sw' else G/('recovery-20261108-v1/'+lang+'-core/results')
  fullsum=SW/'lora-generation-v2/xlsum_sw_v2' if lang=='sw' else (G/('recovery-20261108-v1/ru-summary/xlsum_ru_v1') if lang=='ru' else G/'ar-lora-pipeline-v1/xlsum_ar_v1')
  for variant in plan['variants']:
   run=L/'formal'/lang/variant
   if not (run/'DONE.json').exists():pending.append(lang+'/'+variant);continue
   assert load(run/'DONE.json')['status'].startswith('ALL_FROZEN')
   for target in ['target','en']:
    task=('belebele_swh_Latn_locked' if target=='target' else 'belebele_eng_Latn_locked') if lang=='sw' else 'aya4bml_belebele_'+(lang if target=='target' else 'en')
    a,b=pair(samples(fullhf,task),samples(run/'hf/results',task));assert len(a)==900
    # Candidate likelihood argmax is checked independently of logged ACC.
    for x in a+b:
     v=[float(t[0][0]) for t in x['resps']];assert int(np.argmax(v)==int(x['doc']['correct_answer_num'])-1)==int(x['acc'])
    record(lang,variant,'Belebele '+('target' if target=='target' else 'EN'),'ACC',[x['acc'] for x in a],[x['acc'] for x in b])
   for task in ['sft_v1_gsm8k','sft_v1_ifeval']:
    a,b=pair(samples(fullcore,task),samples(run/'core/results',task))
    if task.endswith('gsm8k'):
     for filt in ['strict-match','flexible-extract']:
      aa=[x for x in a if x['filter']==filt];bb=[x for x in b if x['filter']==filt];assert len(aa)==len(bb)==1319;record(lang,variant,'GSM8K',filt,[x['exact_match'] for x in aa],[x['exact_match'] for x in bb])
    else:
     assert len(a)==541
     for metric in ['prompt_level_strict_acc','inst_level_strict_acc']:
      if metric.startswith('inst'):
       counts=[len(x[metric]) for x in a];assert counts==[len(x[metric]) for x in b] and sum(counts)==834;record(lang,variant,'IFEval',metric,[sum(x[metric]) for x in a],[sum(x[metric]) for x in b],counts)
      else:record(lang,variant,'IFEval',metric,[x[metric] for x in a],[x[metric] for x in b])
   fs=fullsum.with_name(fullsum.name+'-scores');ns=run/'summary-scores'
   if lang=='sw':
    fm=load(fullsum/'manifest.json');nm=load(run/'summary/manifest.json');assert fm['prompt_tokens_sha256']==nm['prompt_tokens_sha256'] and fm['data_manifest_hash']==nm['data_manifest_hash'];a={x['id']:x for x in load(fs/'rouge_per_sample.json')};b={x['id']:x for x in load(ns/'rouge_per_sample.json')};key='ROUGE-Lsum'
   else:
    a={x['id']:x for x in lines(fs/'per_sample_scores.jsonl')};b={x['id']:x for x in lines(ns/'per_sample_scores.jsonl')};key='rougeLsum'
    assert a.keys()==b.keys()
    for k in a:assert a[k]['identity_sha256']==b[k]['identity_sha256']
   assert a.keys()==b.keys() and len(a)=={'sw':987,'ar':4689,'ru':7780}[lang]
   record(lang,variant,'XL-Sum target','ROUGE-Lsum',[a[k][key]/100 for k in sorted(a)],[b[k][key]/100 for k in sorted(a)])
   if lang=='sw':
    fsfiles=list((SW/'lora-harness-v1/gmmlu/results').rglob('samples_global_mmlu_full_sw_*.jsonl'));nsfiles=list((run/'gmmlu/results').rglob('samples_global_mmlu_full_sw_*.jsonl'));assert len(fsfiles)==len(nsfiles)==57;aa=[];bb=[]
    for f in sorted(fsfiles):
     task=f.name.split('samples_',1)[1].rsplit('_',1)[0];a,b=pair(lines(f),samples(run/'gmmlu/results',task));aa.extend(x['acc'] for x in a);bb.extend(x['acc'] for x in b)
    assert len(aa)==14042;record(lang,variant,'Global-MMLU SW','ACC',aa,bb)
   checks.append({'language':lang,'variant':variant,'all_item_identity_checks_passed':True,'complete_primary_tasks':True})
 if rows:
  with (O/'ablation_scores.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 report=['# 实验A：训练后LoRA层级消融','',f'已完成{len(checks)}/18组全量测评；尚未完成：'+(', '.join(pending) or '无')+'。','', '以下变化均相对同语言原Full LoRA；负数表示消融后下降。除摘要ROUGE-Lsum外单位为百分点。不以本表挑选下一轮训练层数。','', '|语言|消融|任务|Base|Full|消融|相对Full变化|配对95%区间|','|---|---|---|---:|---:|---:|---:|---|']
 for x in rows:report.append(f"|{x['language']}|{x['variant']}|{x['task']} {x['metric']}|{x['base']:.2f}|{x['full_lora']:.2f}|{x['ablated']:.2f}|{x['delta_vs_full']:+.2f}|[{x['ci95_low']:+.2f},{x['ci95_high']:+.2f}]|")
 report+=['','## 解释边界','只用seed42，区间未做多重校正；IFEval指令级按提示cluster抽样。保持所有六组，不基于正式测试选k；关闭已训练增量不是部分层重新训练，层间不匹配会影响消融。范数不等于功能重要性，观察关联不证明子空间干扰因果。实验B/C/D尚未运行。']
 (O/'ablation_report.md').write_text('\n'.join(report)+'\n');(O/'evaluation_verification.json').write_text(json.dumps({'complete':not pending,'checks':checks,'pending':pending,'paired_bootstrap':2000},indent=2)+'\n');print(json.dumps({'complete_groups':len(checks),'pending_groups':len(pending),'metric_rows':len(rows)}))
if __name__=='__main__':main()
