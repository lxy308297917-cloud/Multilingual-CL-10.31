"""Read already validated official results; never run a model or choose a checkpoint."""
import argparse,datetime,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.project
assert not a.output.exists();a.output.mkdir(parents=True)
rows=[];sources={};states={}
def read(path):
 sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return json.loads(path.read_text())
def add(method,task,metric,score,n,source,status='TASK_VERIFIED_COMPLETE'):
 assert 0<=score<=100 and n>0
 rows.append({'method':method,'task':task,'metric':metric,'score_0_to_100':score,'denominator':n,'source':str(source),'status':status})
basefile=r/'results/baselines_seed42.json'
for x in read(basefile)['official_metric_rows']:
 add(x['method'],x['task'],x['metric'],x['score_0_to_100'],x.get('instruction_denominator',x['question_or_prompt_count']),basefile)
v2file=r/'results/bench_v2/summary/results.json'
for x in read(v2file)['results']:
 if x['status']!='VERIFIED_COMPLETE' or not x['benchmark'].startswith(('xlsum','flores')):continue
 for metric,score in x['metrics'].items():add(x['method'],x['benchmark'],metric,score,x['samples'],v2file)
runs=[('sst_norefresh',r/'results/sst/formal-norefresh-bench-v1'),('sst_r10',r/'results/sst/formal-r10-bench-v1'),
 ('seq_fft_ha',r/'results/research_my_method/sequential-fft-formal-v1-main'),
 ('base',r/'results/research_my_method/sequential-fft-formal-v1-source-ha-Base'),
 ('full',r/'results/research_my_method/sequential-fft-formal-v1-source-ha-SW-FFT'),
 ('sst_norefresh',r/'results/research_my_method/sequential-sst-norefresh-stage1-ha-v1'),
 ('seq_sst_ha',r/'results/research_my_method/sequential-sst-norefresh-formal-v1-old'),
 ('seq_sst_ha',r/'results/research_my_method/sequential-sst-norefresh-formal-v1-ha')]
for method,folder in runs:
 states[str(folder)]= 'FAILED' if (folder/'FAILED.json').exists() else 'COMPLETE' if (folder/'DONE.json').exists() else 'RUNNING' if (folder/'launch.json').exists() else 'QUEUED'
 if not (folder/'launch.json').exists():continue
 launch=read(folder/'launch.json')
 assert not launch['formal_test_used_for_selection']
 if method.startswith('sst_'):assert launch['actual_method']==method
 if method=='seq_sst_ha':assert launch['actual_method']=='SST-NoRefresh'
 for entry in launch.get('entries',launch.get('commands',[])):
  cmd=entry['command'];task=entry['task']
  key='--output_path' if '--output_path' in cmd else '--output'
  taskdir=Path(cmd[cmd.index(key)+1]).parent;audit=taskdir/'accuracy-audit.json'
  if audit.exists():
   doc=read(audit)
   if 'tasks' in doc:
    assert doc['status']=='validated'
    assert sum(x['count'] for x in doc['tasks'].values())==doc['total_count']
    if 'belebele' in task:
     for name,v in doc['tasks'].items():
      assert v['count']==900 and abs(v['accuracy']-v['correct']/v['count'])<1e-12
      name='belebele_hau_Latn_locked' if 'hau_Latn' in name else name
      add(method,name,'acc,none',100*v['accuracy'],v['count'],audit)
    else:
     assert doc['total_count']==14042
     name={'mmlu':'mmlu_locked','gmmlu':'global_mmlu_full_sw_locked','ha-gmmlu':'global_mmlu_full_ha_locked'}[task]
     add(method,name,'acc,none',100*doc['total_correct']/doc['total_count'],doc['total_count'],audit)
   else:
    assert task=='core'
    for name,metrics in doc.items():
     for metric,v in metrics.items():add(method,name,metric,100*v['rate'],v['n'],audit)
  scorepath=taskdir/'scores/summary.json'
  if scorepath.exists():
   doc=read(scorepath);assert doc['run_status']=='VERIFIED_COMPLETE' and doc['scope']=='full'
   for metric,value in doc['metrics'].items():add(method,task,metric,value,doc['samples'],scorepath)
index={}
for row in rows:
 key=(row['method'],row['task'],row['metric']);assert key not in index,key;index[key]=row
specs=[('SW Belebele ACC','belebele_swh_Latn_locked','acc,none'),('SW Global-MMLU ACC','global_mmlu_full_sw_locked','acc,none'),
 ('SW MGSM strict','sft_v1_mgsm_native_cot_sw','exact_match,strict-match'),('SW MGSM flexible','sft_v1_mgsm_native_cot_sw','exact_match,flexible-extract'),
 ('SW XIFEval prompt strict','sft_v1_xifeval_sw','prompt_level_strict_acc,none'),('SW XL-Sum ROUGE-Lsum','xlsum_sw_v2','ROUGE-Lsum'),
 ('EN Belebele ACC','belebele_eng_Latn_locked','acc,none'),('EN MMLU ACC','mmlu_locked','acc,none'),
 ('EN IFEval prompt strict','sft_v1_ifeval','prompt_level_strict_acc,none'),('EN GSM8K strict','sft_v1_gsm8k','exact_match,strict-match'),('EN GSM8K flexible','sft_v1_gsm8k','exact_match,flexible-extract'),
 ('EN XL-Sum ROUGE-Lsum','xlsum_en_v2','ROUGE-Lsum'),('EN→SW chrF++ full1012','flores200_en_sw_full_devtest_v2','chrF++'),('SW→EN chrF++ full1012','flores200_sw_en_full_devtest_v2','chrF++')]
methods=['base','full','lora','ssu','sst_norefresh','sst_r10']
lines=['# 固定Final正式benchmark汇总','', '只汇入已验证完成任务；RUNNING/PENDING为尚无完整任务结果（包括排队、训练或测评中）。所有ACC为百分数，ROUGE/chrF保留原0–100尺度，不混合平均。', '', '| Benchmark | Base | FFT | LoRA | SSU-Wanda | SST NoRefresh | SST R10 |','|---|---:|---:|---:|---:|---:|---:|']
for label,task,metric in specs:
 vals=[f"{index[m,task,metric]['score_0_to_100']:.2f}" if (m,task,metric) in index else 'RUNNING' for m in methods]
 lines.append('| '+label+' | '+' | '.join(vals)+' |')
lines+=['','## Base→SW FFT→Hausa FFT','', '| Benchmark | Base | SW FFT Final | SW→HA FFT Final |','|---|---:|---:|---:|']
for label,task,metric in [('HA Belebele ACC','belebele_hau_Latn_locked','acc,none'),('HA Global-MMLU ACC','global_mmlu_full_ha_locked','acc,none')]+specs[:5]+specs[6:11]:
 vals=[f"{index[m,task,metric]['score_0_to_100']:.2f}" if (m,task,metric) in index else 'RUNNING' for m in ['base','full','seq_fft_ha']]
 lines.append('| '+label+' | '+' | '.join(vals)+' |')
lines+=['','## Base→SW SST-NoRefresh→Hausa SST-NoRefresh','', '| Benchmark | Base | SW SST Final | SW→HA SST Final |','|---|---:|---:|---:|']
for label,task,metric in [('HA Belebele ACC','belebele_hau_Latn_locked','acc,none'),('HA Global-MMLU ACC','global_mmlu_full_ha_locked','acc,none')]+specs[:5]+specs[6:11]:
 vals=[f"{index[m,task,metric]['score_0_to_100']:.2f}" if (m,task,metric) in index else 'PENDING' for m in ['base','sst_norefresh','seq_sst_ha']]
 lines.append('| '+label+' | '+' | '.join(vals)+' |')
lines+=['','开发114题与有限FP32干预另表，不能与本表相减。SST两组full_determinism与旧FFT执行差异保留；所有方法均固定400步Final，无EN早停门槛。Hausa没有当前锁定版本的官方BenchMAX规则任务，未冒称已测。','', '代码benchmark四模型已发布，SST代码分数NOT_RUN，本轮固定SST主要清单不含该新增任务。']
report={'snapshot_at':datetime.datetime.now().astimezone().isoformat(),'status':'PARTIAL_FIXED_TASKS' if any(x!='COMPLETE' for x in states.values()) else 'COMPLETE',
 'rows':rows,'run_states':states,'sources_sha256':sources,'no_test_based_selection':True,'no_mixed_metric_average':True}
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'tables.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'rows':len(rows),'states':states}))
