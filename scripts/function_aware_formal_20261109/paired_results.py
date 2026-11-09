"""Independently verify per-item scores and align frozen inputs before paired CIs."""
from pathlib import Path
import argparse,collections,csv,hashlib,json
import numpy as np
R=Path('/path/to/project');Q=R/'research/function_aware_formal_20261109'
L=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109/evaluation')
OLD=Path('/path/to/logs/qwen3-multilingual/aya-4b-20261105')
MULTI=Path('/path/to/logs/qwen3-multilingual/aya-4b-multilang-20261106')
GEN=Path('/path/to/logs/qwen3-multilingual/aya-4b-generalization-20261107')
P2=Path('/path/to/logs/qwen3-multilingual/functional-representation-20261109/p2/evaluation')
RB=Path('/path/to/logs/qwen3-multilingual/representation-benchmarks-20261109')
TASKS=['target_reading','target_summary','target_knowledge','target_xnli','en_gsm8k','en_ifeval','en_mmlu','en_code','en_reading','en_xnli']
def load(p):return json.loads(p.read_text())
def rows(p):
 with Path(p).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def only(root,pattern):
 paths=list(root.rglob(pattern));assert len(paths)==1,(root,pattern,len(paths));return paths[0]
def signature(x):return tuple(x[k] for k in ['doc_hash','prompt_hash','target_hash'])
def mc(root,pattern):
 vector={};source=[]
 samplefiles=sorted(root.rglob('samples_'+pattern+'_*.jsonl'));assert samplefiles,(root,pattern)
 result=load(only(root,'results_*.json'))['results']
 for path in samplefiles:
  task=path.name.removeprefix('samples_').rsplit('_',1)[0];values=[]
  for row in rows(path):
   assert row['filter']=='none'
   scores=[float(x[0]) if isinstance(x,(tuple,list)) else float(x) for x in row['filtered_resps']]
   assert np.isfinite(scores).all();pred=int(np.argmax(scores));doc=row['doc']
   if 'correct_answer_num' in doc:gold=int(doc['correct_answer_num'])-1
   elif 'label' in doc:gold=int(doc['label'])
   else:gold='ABCD'.index(row['target']) if isinstance(row['target'],str) and row['target'] in 'ABCD' else int(row['target'])
   value=int(pred==gold);assert value==row['acc'];values.append(value)
   key=(task,str(row['doc_id'])) if 'mmlu' in task else str(row['doc_id'])
   cluster=doc['premise_cluster_sha256'] if 'premise_cluster_sha256' in doc else digest(doc['flores_passage']) if 'flores_passage' in doc else str(key)
   assert key not in vector;vector[key]=dict(identity=signature(row),value=value*100,cluster=cluster)
  assert abs(np.mean(values)-result[task]['acc,none'])<1e-10,(root,task)
  source.append(dict(path=str(path),sha256=sha(path),samples=len(values)))
 return vector,source

def core(root,task):
 name='sft_v1_gsm8k' if task=='en_gsm8k' else 'sft_v1_ifeval';path=only(root,'samples_'+name+'_*.jsonl');result=load(only(root,'results_*.json'))['results'][name];v={}
 for row in rows(path):
  if task=='en_gsm8k':
   if row['filter']!='flexible-extract':continue
   value=float(row['exact_match'])
  else:
   assert row['filter']=='none';value=float(row['prompt_level_strict_acc']);assert bool(value)==all(row['inst_level_strict_acc'])
  key=str(row['doc_id']);assert key not in v;v[key]=dict(identity=signature(row),value=value*100,cluster=key)
 expected=1319 if task=='en_gsm8k' else 541;metric='exact_match,flexible-extract' if task=='en_gsm8k' else 'prompt_level_strict_acc,none'
 assert len(v)==expected and abs(np.mean([x['value'] for x in v.values()])/100-result[metric])<1e-10
 return v,[dict(path=str(path),sha256=sha(path),samples=len(v))]

def summary(run,scored,lang):
 manifest=load(run/'manifest.json');count={'sw':987,'ar':4689,'ru':7780}[lang]
 scorefile=scored/('summary.json' if lang=='sw' else 'scores.json');aggregate=load(scorefile)
 assert aggregate['samples']==count and sha(run/'predictions.jsonl')==aggregate['predictions_sha256']
 if lang=='sw':
  xs=load(scored/'rouge_per_sample.json');identity=(manifest['prompt_tokens_sha256'],manifest['data_manifest_hash'])
  v={str(x['id']):dict(identity=identity,value=x['ROUGE-Lsum'],cluster=str(x['id'])) for x in xs}
 else:
  xs=list(rows(scored/'per_sample_scores.jsonl'));v={str(x['id']):dict(identity=(x['identity_sha256'],),value=x['rougeLsum'],cluster=str(x['id'])) for x in xs}
 assert len(xs)==len(v)==count and abs(np.mean([x['value'] for x in v.values()])-aggregate['metrics']['ROUGE-Lsum'])<1e-7
 return v,[dict(path=str(scorefile),sha256=sha(scorefile),samples=count)]

def code(root):
 execution=root/'execution';aggregate=load(execution/'scores.json');xs=load(execution/'model_evaluation-results.json');assert aggregate['status']=='COMPLETE' and len(xs)==164
 assert all(x['infrastructure_status']=='COMPLETE' for x in xs)
 source=aggregate['executor_identity']['raw_sha256'];v={}
 for row in xs:
  assert not row['plus_pass'] or row['base_pass'];key=row['task_id'];assert key not in v
  v[key]=dict(identity=(source,key),value=int(row['plus_pass'])*100,cluster=key)
 assert len(v)==164 and sum(x['value'] for x in v.values())==aggregate['plus_correct']*100
 return v,[dict(path=str(execution/'scores.json'),sha256=sha(execution/'scores.json'),samples=164)]

def locate(lang,method,task):
 """Return a completed reader call or None; never read an unfinished score as zero."""
 root=L/(lang+'_'+method)
 if method=='Base' or (lang=='sw' and method=='all') or (lang=='sw' and method=='HistoricalFull36'):
  base=method=='Base';p2name='all_layer' if method=='all' else 'full_feature'
  if task=='en_mmlu':
   folder=Path('/path/to/logs/qwen3-multilingual/fft-representation-20261109/evaluation/p2_full36/mmlu') if method=='HistoricalFull36' else L/('sw_Base' if base else 'sw_all')/'mmlu'
   if base:return mc(OLD/'base-harness-v1/mmlu/results','mmlu_*')
   return mc(folder,'mmlu_*') if (folder/'DONE.json').exists() else None
  if task=='target_knowledge' and lang!='sw':return mc(root/'gmmlu','global_mmlu_full_'+lang+'_*') if (root/'gmmlu/DONE.json').exists() else None
  if task=='en_code':return code(RB/('formal-'+('base' if base else p2name)+'-code'))
  if task in ['en_xnli','target_xnli'] and (task=='en_xnli' or lang=='sw'):
   which='en' if task=='en_xnli' else 'sw';return mc(RB/('formal-'+('base' if base else p2name)+'-xnli')/'results','rb_xnli_'+which+'_formal_primary')
  if task in ['en_gsm8k','en_ifeval']:return core(OLD/'base-generation-v2/core/results' if base else P2/p2name/'core/results',task)
  if task=='en_reading' or (task=='target_reading' and lang=='sw'):
   name='belebele_eng_Latn_locked' if task=='en_reading' else 'belebele_swh_Latn_locked'
   return mc(OLD/'base-harness-v1/belebele/results' if base else P2/p2name/'hf/results',name)
  if task=='target_knowledge':return mc(OLD/'base-harness-v1/gmmlu/results' if base else P2/p2name/'gmmlu/results','global_mmlu_full_sw_*')
  if task in ['target_reading','target_xnli']:
   folder=MULTI/('base-hf-v1' if lang=='ar' else 'base-ru-hf-v1')/'results';name='aya4bml_'+('belebele' if task=='target_reading' else 'xnli')+'_'+lang
   return mc(folder,name)
  if task=='target_summary':
   if lang=='sw':
    folder=OLD/'base-generation-v2/xlsum_sw_v2' if base else P2/p2name/'summary';scores=folder.with_name(folder.name+'-scores') if base else P2/p2name/'summary-scores'
   else:folder=GEN/('base-'+lang+'-summary-v1')/('xlsum_'+lang+'_v1');scores=folder.with_name(folder.name+'-scores')
   return summary(folder,scores,lang)
  raise AssertionError((lang,method,task))
 stage='reading' if task in ['target_reading','en_reading','target_xnli','en_xnli'] else 'summary' if task=='target_summary' else 'gmmlu' if task=='target_knowledge' else 'mmlu' if task=='en_mmlu' else 'core' if task in ['en_gsm8k','en_ifeval'] else 'code'
 folder=root/stage
 if not (folder/'DONE.json').exists():return None
 if stage=='reading':
  name={'target_reading':'belebele_swh_Latn_locked' if lang=='sw' else 'aya4bml_belebele_'+lang,'en_reading':'belebele_eng_Latn_locked','target_xnli':'rb_xnli_sw_formal_primary' if lang=='sw' else 'aya4bml_xnli_'+lang,'en_xnli':'rb_xnli_en_formal_primary'}[task]
  return mc(folder,name)
 if stage in ['mmlu','gmmlu']:return mc(folder,'mmlu_*' if stage=='mmlu' else 'global_mmlu_full_'+lang+'_*')
 if stage=='core':return core(folder,task)
 if stage=='summary':return summary(folder/'generation',folder/'scores',lang)
 return code(folder)

def paired(a,b,seed):
 assert a.keys()==b.keys(),'Different task IDs'
 keys=sorted(a,key=str);assert all(a[k]['identity']==b[k]['identity'] and a[k]['cluster']==b[k]['cluster'] for k in keys),'Prompt/target/task identity changed'
 av=np.array([a[k]['value'] for k in keys]);bv=np.array([b[k]['value'] for k in keys]);groups=[str(a[k]['cluster']) for k in keys]
 unique,inverse=np.unique(groups,return_inverse=True);n=len(unique);sums=np.bincount(inverse,weights=bv-av,minlength=n);sizes=np.bincount(inverse,minlength=n)
 rng=np.random.default_rng(seed);bootstrap=[]
 for _ in range(40):
  idx=rng.integers(n,size=(50,n));bootstrap.extend(sums[idx].sum(1)/sizes[idx].sum(1))
 low,high=np.quantile(bootstrap,[.025,.975]);return dict(delta=float((bv-av).mean()),ci95_low=float(low),ci95_high=float(high),samples=len(keys),clusters=n,resamples=2000,task_identity_sha256=digest([(str(k),a[k]['identity'],a[k]['cluster']) for k in keys]))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args();vectors={};provenance=[];missing=[];absolute=[]
 main_scores={(x['language'],x['method'],x['task']):float(x['score']) for x in csv.DictReader((Q/'scores_long.csv').open()) if x['score']!=''}
 for lang in ['sw','ar','ru']:
  for method in ['Base','all','full','kd','random','function']+(['HistoricalFull36'] if lang=='sw' else []):
   for task in TASKS:
    found=locate(lang,method,task)
    if found is None:missing.append((lang,method,task));continue
    vector,files=found;assert all(0<=x['value']<=100 for x in vector.values());score=float(np.mean([x['value'] for x in vector.values()]));key=(lang,method,task)
    assert key in main_scores and abs(score-main_scores[key])<1e-7,(key,score,main_scores.get(key))
    vectors[key]=vector;absolute.append(dict(language=lang,method=method,task=task,score=score,samples=len(vector),clusters=len({x['cluster'] for x in vector.values()})))
    provenance.append(dict(language=lang,method=method,task=task,files=files))
 numeric=Q/'numeric';numeric.mkdir(exist_ok=True)
 for lang in ['sw','ar','ru']:
  for task in TASKS:
   models=[m for m in ['Base','all','full','kd','random','function','HistoricalFull36'] if (lang,m,task) in vectors]
   if not models:continue
   anchor=vectors[lang,models[0],task];keys=sorted(anchor,key=str)
   for method in models:
    other=vectors[lang,method,task]
    assert anchor.keys()==other.keys() and all(anchor[k]['identity']==other[k]['identity'] and anchor[k]['cluster']==other[k]['cluster'] for k in keys),'Numeric matrix has mismatched input identities'
   _,inverse=np.unique([str(anchor[k]['cluster']) for k in keys],return_inverse=True)
   matrix=dict(language=lang,task=task,models=models,item_id_sha256=[digest(str(k)) for k in keys],unit='ROUGE points' if task=='target_summary' else 'percent',samples=len(keys),clusters=len(set(inverse.tolist())),cluster_index=inverse.tolist(),values=[[vectors[lang,m,task][k]['value'] for m in models] for k in keys],ordered_input_identity_sha256=digest([(str(k),anchor[k]['identity']) for k in keys]),scope='Numeric sufficient statistics only; no task text, references, generated answers, paths or checkpoint weights',purpose='Recompute paired benchmark means and cluster bootstrap CIs; no model selection')
   (numeric/(lang+'_'+task+'.json')).write_text(json.dumps(matrix,separators=(',',':'))+'\n')
 contrasts=[('Base','all'),('Base','full'),('Base','kd'),('Base','random'),('Base','function'),('all','full'),('all','kd'),('all','random'),('all','function'),('full','function'),('random','function'),('kd','function')]
 intervals=[]
 for langindex,lang in enumerate(['sw','ar','ru']):
  for ci,(a,b) in enumerate(contrasts+([('HistoricalFull36','function')] if lang=='sw' else [])):
   for ti,task in enumerate(TASKS):
    if (lang,a,task) not in vectors or (lang,b,task) not in vectors:continue
    intervals.append(dict(language=lang,reference=a,comparison=b,task=task,bootstrap_unit='premise' if 'xnli' in task else 'passage' if 'reading' in task else 'article' if task=='target_summary' else 'task',**paired(vectors[lang,a,task],vectors[lang,b,task],np.random.SeedSequence([42,langindex,ci,ti]))))
 if not args.partial:assert len([k for k in missing if k[1]!='HistoricalFull36'])==0,missing
 for name,data in [('audited_scores.csv',absolute),('paired_changes.csv',intervals)]:
  if not data:continue
  with (Q/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (Q/'score_source_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 report=dict(status='INCOMPLETE_FORMAL_RESULTS' if any(x[1]!='HistoricalFull36' for x in missing) else 'ALL_MAIN_CELLS_RAW_AUDITED',verified_cells=len(vectors),main_cells_verified=sum(k[1]!='HistoricalFull36' for k in vectors),historical_cells_verified=sum(k[1]=='HistoricalFull36' for k in vectors),required_main_cells=180,missing=missing,paired_comparisons=len(intervals),all_acc_argmax_recomputed=True,all_aggregates_match_items=True,paired_identity_verified=True,bootstrap_resamples=2000,bootstrap_note='Fixed model/benchmark uncertainty only, not training-seed robustness; no multiple-comparison correction; no method selection',question_text_publication=False)
 (Q/'score_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='missing'}))
if __name__=='__main__':main()
