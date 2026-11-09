"""Build readable tables, descriptive model associations, and measured cost records."""
from pathlib import Path
import csv,datetime,json,math
import numpy as np
R=Path('/data0/users/lixiaoyang/10.31/multilingual-research')
N=R/'research/representation_benchmarks_20261109'
L=Path('/data0/users/lixiaoyang/logs/qwen3-multilingual/representation-benchmarks-20261109')
MODELS=['Base','all_layer','full_feature','ce_replay']
def read(p):return list(csv.DictReader(p.open()))
def write(p,xs):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(xs[0]));w.writeheader();w.writerows(xs)
def ranks(x):
 x=np.asarray(x);out=np.zeros(len(x));order=np.argsort(x)
 for v in set(x.tolist()):
  positions=np.flatnonzero(x[order]==v);out[order[positions]]=positions.mean()+1
 return out
def main():
 assert (N/'benchmarks/DONE.json').exists() and (N/'mechanism/DONE.json').exists()
 scores=read(N/'complete_scores.csv');new=read(N/'benchmarks/new_scores.csv');pairs=read(N/'benchmarks/paired_changes.csv')
 lines=['# 四模型完整固定端点成绩','', '共同 seed42、batch2、Final400；Base未适配。摘要为ROUGE-Lsum，其余为百分比。主要表不含XNLI辅助轮换，不混合单位求平均。','', '|任务／指标|Base|普通LoRA|全表示保持|CE回放|','|---|---:|---:|---:|---:|']
 primary=[];aux=[]
 for x in scores:
  (aux if x['metric'] in ['strict-match','inst_level_strict_acc'] else primary).append(x)
 assert len(primary)==9
 for x in primary:lines.append('|'+x['task']+' / '+x['metric']+'|'+'|'.join(f"{float(x[m]):.2f}" for m in MODELS)+'|')
 lines+=['','## 辅助指标','', '|任务／指标|Base|普通LoRA|全表示保持|CE回放|','|---|---:|---:|---:|---:|']
 for x in aux:lines.append('|'+x['task']+' / '+x['metric']+'|'+'|'.join(f"{float(x[m]):.2f}" for m in MODELS)+'|')
 human=[x for x in new if x['task']=='HumanEval'];lines+=['|HumanEval 原始测试 pass@1|'+'|'.join(f"{float(next(x['score'] for x in human if x['model']==m)):.2f}" for m in MODELS)+'|','', '## XNLI：固定标签轮换诊断','', '不是替代主要协议的更优分数。历史Base已有选项偏好；本轮相同规则用于全部模型。','', '|模型|SW主要|SW轮换|EN主要|EN轮换|','|---|---:|---:|---:|---:|']
 for m in MODELS:lines.append('|'+m+'|'+'|'.join(f"{float(next(x['score'] for x in new if x['model']==m and x['task']=='XNLI '+lang and x['protocol']==kind)):.2f}" for lang in ['SW','EN'] for kind in ['primary','rotated'])+'|')
 lines+=['','## 新增任务配对区间','', '区间单位为百分点；XNLI以共享premise聚类重采样，SW1669群、EN1670群；代码以164道题重采样。均为固定模型的2000次配对bootstrap，不是训练种子置信区间。','', '|任务|比较（后者减前者）|差值|配对95%区间|旧对→新错|旧错→新对|','|---|---|---:|---|---:|---:|']
 for x in pairs:
  if x['protocol'] not in ['primary','greedy_n1_isolated_evalplus'] or x['task']=='HumanEval':continue
  lines.append(f"|{x['task']}|{x['reference']} → {x['comparison']}|{float(x['delta_percentage_points']):+.2f}|[{float(x['paired95_low']):+.2f}, {float(x['paired95_high']):+.2f}]|{x['base_correct_adapted_wrong']}|{x['base_wrong_adapted_correct']}|")
 lines+=['','## 完整材料','', '- [新增绝对分数](benchmarks/new_scores.csv)、[全部配对区间](benchmarks/paired_changes.csv)、[XNLI类别与选项分布](benchmarks/xnli_label_bias.json)、[代码执行审计](benchmarks/code_execution_audit.json)。','- 旧任务配对结果及57学科明细保持在此前[P2目录](../functional_representation_20261109/p2/results.md)；没有覆盖历史数据。','- 翻译本轮未补测；新增推断/代码成绩不能证明翻译退化已修复。']
 (N/'results.md').write_text('\n'.join(lines)+'\n')
 summary=read(N/'mechanism/model_level_summary.csv');signal=[]
 sim=read(N/'mechanism/cross_language_layers.csv')
 for m in MODELS:
  x=next(x for x in summary if x['model']==m)
  row=dict(model=m,old_check_squared_l2_mean=float(np.mean([float(x['check_'+c+'_l2_mean36']) for c in ['math','instruction']])),old_check_covariance_fro_mean=float(np.mean([float(x['check_'+c+'_sample_cov_fro_mean36']) for c in ['math','instruction']])),effective_update_norm=float(x['effective_update_frobenius']))
  for c in ['raw_content_mean','chat_content_tokens_mean']:
   vals=[abs(float(s['delta_mean_vs_base'])) for s in sim if s['model']==m and s['condition']==c and 1<=int(s['layer'])<=36]
   row[c+'_absolute_mean_layer_delta']=float(np.mean(vals))
  signal.append(row)
 write(N/'mechanism/model_signals.csv',signal);associations=[]
 for name in [k for k in signal[0] if k!='model']:
  a=np.array([s[name] for s in signal]);ar=ranks(a)
  for task in primary:
   b=np.array([float(task[m]) for m in MODELS]);br=ranks(b)
   r=float(np.corrcoef(ar,br)[0,1]) if np.std(ar)>0 and np.std(br)>0 else None
   associations.append(dict(signal=name,task=task['task'],metric=task['metric'],descriptive_spearman_four_conditions=r,n_model_conditions=4,inferential_p_value='NOT_IDENTIFIABLE_AS_INDEPENDENT_MODEL_REPLICATES',caveat='same Base/seed/target training; Base anchors zero drift; no independent added-value or causal claim'))
 write(N/'mechanism/descriptive_model_associations.csv',associations)
 jobs=[]
 for d in L.iterdir():
  f=d/'inference.log.command.json' if d.is_dir() else None
  if f is None or not f.exists() or not (d/'DONE.json').exists():continue
  c=json.loads(f.read_text());state=json.loads((d/'state.json').read_text());done=json.loads((d/'DONE.json').read_text());start=datetime.datetime.fromisoformat(c['started_at']);end=datetime.datetime.fromisoformat(state['updated_at'])
  jobs.append(dict(job=d.name,stage=done['stage'],scope=done['scope'],model=done['model'],command_to_completion_seconds=(end-start).total_seconds(),including_queue_seconds=done['elapsed_seconds_including_wait']))
 mechanism=json.loads((N/'mechanism/DONE.json').read_text())['sources'];code=[]
 for m in MODELS:
  p=L/('formal-'+m.lower()+'-code');s=json.loads((p/'execution/scores.json').read_text());c=json.loads((p/'generation/cost.json').read_text());xs=json.loads((p/'execution/model_evaluation-results.json').read_text());code.append(dict(model=m,generated_tokens=c['generated_tokens'],input_tokens=c['input_tokens'],generation_seconds_excluding_initial_prompt_hash_preparation=c['elapsed_seconds'],model_sandbox_serial_coordinator_seconds=s['seconds_model_execution'],sum_isolated_task_wall_seconds=sum(x['execution']['wall_seconds'] for x in xs)))
 cost=dict(new_training_jobs=0,account_gpu_limit=4,formal_xnli_primary_predictions=40080,formal_xnli_rotation_predictions=40080,formal_code_generations=656,official_canonical_isolated_checks=164,formal_model_isolated_checks=656,parallel_probe_pairs=300,old_english_probe_rows=256,benchmark_jobs=jobs,representation_models=mechanism,code=code,note='Command-to-completion includes model init/cleanup and is NOT pure kernel time. Queue waits separate. Mechanism process seconds include model load. Do not equate sample tasks or layer count with independent model conditions; no new training cost.')
 (N/'cost_summary.json').write_text(json.dumps(cost,indent=2)+'\n');print('Nine primary rows, two auxiliary rows, complete paired contrasts and measured costs written')
if __name__=='__main__':main()
