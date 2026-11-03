"""Saved official per-item contrasts, verifying docs/prompts and aggregates first."""
import argparse,collections,hashlib,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--reference-method',choices=['base','full'],default='base');p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.project
assert not a.output.exists();a.output.mkdir(parents=True)
sources={}
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def read(f):sources[str(f)]=sha(f);return json.loads(f.read_text())
old=read(r/'results/baselines_seed42.json')['official_metric_rows'];launch=read(a.run/'launch.json')
assert not launch['formal_test_used_for_selection']
reports=[];missing=[]
for entry in launch.get('entries',launch.get('commands',[])):
 cmd=entry['command'];task=entry['task']
 if task not in ['core','belebele','mmlu','gmmlu']:continue
 folder=Path(cmd[cmd.index('--output_path')+1]).parent
 if not (folder/'accuracy-audit.json').exists():missing.append(task);continue
 read(folder/'accuracy-audit.json')
 paths=list(folder.glob('results/**/results_*.json'));assert len(paths)==1
 targetfile=paths[0];targetresult=read(targetfile)['results']
 if task=='core':lookup='sft_v1_ifeval'
 else:lookup={'belebele':'belebele_swh_Latn_locked','mmlu':'mmlu_locked','gmmlu':'global_mmlu_full_sw_locked'}[task]
 references={x['official_result_file'] for x in old if x['method']==a.reference_method and x['task']==lookup};assert len(references)==1
 oldfile=Path(next(iter(references)));oldresult=read(oldfile)['results']
 # All concrete tasks have samples; groups such as MMLU have no sample file.
 names=sorted(k for k in targetresult if list(targetfile.parent.glob('samples_'+k+'_*.jsonl')))
 grouped=collections.defaultdict(list)
 for name in names:
  tpaths=list(targetfile.parent.glob('samples_'+name+'_*.jsonl'));bpaths=list(oldfile.parent.glob('samples_'+name+'_*.jsonl'));assert len(tpaths)==len(bpaths)==1
  sources[str(tpaths[0])]=sha(tpaths[0]);sources[str(bpaths[0])]=sha(bpaths[0])
  target=[json.loads(x) for x in tpaths[0].open()];base=[json.loads(x) for x in bpaths[0].open()]
  ref={(x['doc_id'],x['filter']):x for x in base};assert len(ref)==len(base)
  seen=set()
  for row in target:
   key=(row['doc_id'],row['filter']);assert key in ref and key not in seen;seen.add(key);b=ref[key]
   assert row['doc']==b['doc'] and row['arguments']==b['arguments'],'Actual doc/prompt/options differ: '+name
   for metric in row['metrics']:
    before=b[metric] if isinstance(b[metric],list) else [b[metric]]
    after=row[metric] if isinstance(row[metric],list) else [row[metric]]
    assert len(before)==len(after) and before and all(v in (0,1) for v in before+after)
    # Every prompt's instruction cluster stays together during bootstrap.
    bucket=name if task in ['core','belebele'] else lookup
    grouped[bucket,metric,row['filter']].append((sum(before),sum(after),len(before),name))
  assert seen==set(ref)
 for (name,metric,label),pairs in sorted(grouped.items()):
  x=np.array([v[:3] for v in pairs],dtype=float);b=x[:,0];t=x[:,1];n=x[:,2]
  oldrate=float(b.sum()/n.sum());rate=float(t.sum()/n.sum())
  official_name=name;key=metric+','+label
  assert abs(oldrate-oldresult[official_name][key])<1e-10,(name,'reference aggregate')
  assert abs(rate-targetresult[official_name][key])<1e-10,(name,'target aggregate')
  rng=np.random.default_rng(42);delta=t-b
  boot=[]
  for _ in range(2000):
   ids=rng.integers(0,len(x),len(x));boot.append(100*delta[ids].sum()/n[ids].sum())
  lo,hi=np.percentile(boot,[2.5,97.5])
  reports.append({'task':name,'metric':key,'reference_method':a.reference_method,'reference_correct':int(b.sum()),'target_correct':int(t.sum()),'denominator':int(n.sum()),'prompts':len(x),
   'reference_score':100*oldrate,'target_score':100*rate,'delta_pp':100*(rate-oldrate),'ci95_low':float(lo),'ci95_high':float(hi),'more_correct_prompts':int((delta>0).sum()),'fewer_correct_prompts':int((delta<0).sum()),'same_correct_prompts':int((delta==0).sum()),'actual_doc_and_prompt_exact':True,'repeats':2000,'seed':42,'unit':'paired original prompts; instruction clusters preserved; MMLU pooled items','multiple_comparisons':'unadjusted descriptive intervals, no selection'})
out={'status':'VERIFIED_COMPLETED_TASKS_ONLY','method':launch.get('actual_method'),'reference_method':a.reference_method,'missing_tasks':missing,'contrasts':reports,'sources_sha256':sources,'scoring_modified':False,'test_used_for_selection':False}
(a.output/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# 已完成正式任务的配对变化','', '| Task | Metric | Reference | Target | Δ pp | 95% paired interval |','|---|---|---:|---:|---:|---:|']
for x in reports:lines.append('| %s | %s | %.2f | %.2f | %+.2f | [%+.2f, %+.2f] |'%(x['task'],x['metric'],x['reference_score'],x['target_score'],x['delta_pp'],x['ci95_low'],x['ci95_high']))
lines+=['','所有实际doc和prompt/options逐条相同。区间为描述性，未作多重校正，不用于挑任务或选模型。','未完成任务：'+', '.join(missing)]
(a.output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'contrasts':len(reports),'missing_tasks':missing}))
