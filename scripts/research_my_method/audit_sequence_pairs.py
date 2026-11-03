"""Pair completed phase records, preserving actual prompts and IFEval clusters."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--reference-record',type=Path,nargs='*',default=[]);p.add_argument('--legacy-method',choices=['base','full']);p.add_argument('--target-record',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);sources={}
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def read(f):sources[str(f)]=sha(f);return json.loads(f.read_text())
def index(records):
 result={}
 for folder in records:
  assert (folder/'DONE.json').exists() and not (folder/'FAILED.json').exists(),str(folder)+' is incomplete'
  receipt=read(folder/'DONE.json');assert not receipt['formal_test_used_for_selection'];launch=read(folder/'launch.json')
  for entry in launch.get('entries',launch.get('commands',[])):
   task=entry['task']
   if task not in ['core','belebele','mmlu','gmmlu','ha-belebele','ha-gmmlu']:continue
   cmd=entry['command'];taskdir=Path(cmd[cmd.index('--output_path')+1]).parent;read(taskdir/'accuracy-audit.json')
   files=list(taskdir.glob('results/**/results_*.json'));assert len(files)==1 and task not in result
   result[task]=files[0]
 return result
reference=index(a.reference_record)
if a.legacy_method:
 old=read(a.project/'results/baselines_seed42.json')['official_metric_rows']
 for group,key in [('core','sft_v1_ifeval'),('belebele','belebele_swh_Latn_locked'),('mmlu','mmlu_locked'),('gmmlu','global_mmlu_full_sw_locked')]:
  files={x['official_result_file'] for x in old if x['method']==a.legacy_method and x['task']==key};assert len(files)==1 and group not in reference
  reference[group]=Path(next(iter(files)))
target=index(a.target_record);assert target and target.keys()<=reference.keys()
contrasts=[]
for kind,tf in sorted(target.items()):
 bf=reference[kind];ts=read(tf)['results'];bs=read(bf)['results'];names=sorted(k for k in ts if list(tf.parent.glob('samples_'+k+'_*.jsonl')));grouped={}
 for name in names:
  tpaths=list(tf.parent.glob('samples_'+name+'_*.jsonl'));bpaths=list(bf.parent.glob('samples_'+name+'_*.jsonl'));assert len(tpaths)==len(bpaths)==1
  sources[str(tpaths[0])]=sha(tpaths[0]);sources[str(bpaths[0])]=sha(bpaths[0]);rows=[json.loads(x) for x in tpaths[0].open()];baseline=[json.loads(x) for x in bpaths[0].open()];ref={(x['doc_id'],x['filter']):x for x in baseline};assert len(ref)==len(baseline);seen=set()
  for row in rows:
   key=(row['doc_id'],row['filter']);assert key not in seen and key in ref;seen.add(key);old=ref[key]
   assert old['doc']==row['doc'] and old['arguments']==row['arguments'],(kind,name,'different actual doc/prompt')
   for metric in row['metrics']:
    before=old[metric] if isinstance(old[metric],list) else [old[metric]];after=row[metric] if isinstance(row[metric],list) else [row[metric]];assert before and len(before)==len(after) and all(x in (0,1) for x in before+after)
    namegroup=name if kind in ['core','belebele','ha-belebele'] else {'mmlu':'mmlu_locked','gmmlu':'global_mmlu_full_sw_locked','ha-gmmlu':'global_mmlu_full_ha_locked'}[kind]
    grouped.setdefault((namegroup,metric,row['filter']),[]).append((sum(before),sum(after),len(before)))
  assert seen==set(ref)
 for (name,metric,label),vals in sorted(grouped.items()):
  x=np.array(vals,dtype=float);before=x[:,0];after=x[:,1];counts=x[:,2];br=before.sum()/counts.sum();tr=after.sum()/counts.sum();key=metric+','+label
  assert abs(bs[name][key]-br)<1e-10 and abs(ts[name][key]-tr)<1e-10
  rng=np.random.default_rng(42);delta=after-before;boot=[]
  for _ in range(2000):
   ids=rng.integers(0,len(vals),len(vals));boot.append(100*delta[ids].sum()/counts[ids].sum())
  low,high=np.percentile(boot,[2.5,97.5]);contrasts.append({'task_kind':kind,'task':name,'metric':key,'reference_correct':int(before.sum()),'target_correct':int(after.sum()),'denominator':int(counts.sum()),'prompt_count':len(vals),'reference_score':float(br*100),'target_score':float(tr*100),'delta_pp':float((tr-br)*100),'ci95_low':float(low),'ci95_high':float(high),'more_correct_prompts':int((delta>0).sum()),'fewer_correct_prompts':int((delta<0).sum()),'unchanged_prompts':int((delta==0).sum())})
result={'status':'VERIFIED_COMPLETE_PHASE_COMPARISON','reference_records':[str(x) for x in a.reference_record],'legacy_method':a.legacy_method,'target_records':[str(x) for x in a.target_record],'all_actual_doc_and_prompt_exact':True,'contrasts':contrasts,'sources_sha256':sources,'analysis_code_sha256':sha(Path(__file__)),'bootstrap':{'repeats':2000,'seed':42,'unit':'paired prompts; instruction clusters kept together, knowledge pooled items','multiple_comparisons':'unadjusted descriptive intervals'},'test_used_for_selection':False}
(a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# 固定阶段之间的配对正式变化','','| Task | Metric | Before | After | Δ pp | 95% paired interval |','|---|---|---:|---:|---:|---:|']
for v in contrasts:lines.append('| %s | %s | %.2f | %.2f | %+.2f | [%+.2f, %+.2f] |'%(v['task'],v['metric'],v['reference_score'],v['target_score'],v['delta_pp'],v['ci95_low'],v['ci95_high']))
lines+=['','实际doc、prompt与选项逐项相同。只比较完整固定Final的正式任务；无选模、无代理替换。区间未多重校正，保持IFEval提示内部指令cluster。']
(a.output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'task_groups':len(target),'contrasts':len(contrasts),'all_doc_prompt_exact':True}))
