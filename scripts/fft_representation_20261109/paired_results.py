"""Raw FFT score audit and paired benchmark uncertainty; no training or selection."""
from pathlib import Path
import argparse,csv,importlib.util,json,sys
import numpy as np
R=Path('/path/to/project')
Q=R/'research/fft_representation_20261109'
L=Path('/path/to/logs/qwen3-multilingual/fft-representation-20261109/evaluation')
spec=importlib.util.spec_from_file_location('raw_audit',R/'scripts/function_aware_formal_20261109/paired_results.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
sys.path.insert(0,str(Path(__file__).parent));from collect_results import PROFILES,TASKS
OLD=audit.OLD;P2=audit.P2;RB=audit.RB
TASK_MAP={'SW Belebele':'target_reading','SW XL-Sum':'target_summary','SW Global-MMLU':'target_knowledge','SW XNLI':'target_xnli','EN Belebele':'en_reading','EN XNLI':'en_xnli','EN MMLU':'en_mmlu','GSM8K flexible':'en_gsm8k','IFEval prompt strict':'en_ifeval','HumanEval+':'en_code'}

def canonical_mc(root,pattern,xnli=False):
 vector,source=audit.mc(root,pattern)
 if xnli:
  # Old/new records differ only in unused rotation metadata. Verify effective
  # prompt, target, question ID and labels instead of discarding identity checks.
  for file in root.rglob('samples_'+pattern+'_*.jsonl'):
   for row in audit.rows(file):
    key=str(row['doc_id']);doc=row['doc']
    assert int(doc['label'])==int(row['target'])
    vector[key]['identity']=(row['prompt_hash'],row['target_hash'],audit.digest([doc['id'],doc['prompt'],doc['label']]))
 return vector,source

def stage_reader(folder,stage,task):
 if stage=='reading':
  name={'SW Belebele':'belebele_swh_Latn_locked','EN Belebele':'belebele_eng_Latn_locked','SW XNLI':'rb_xnli_sw_formal_primary','EN XNLI':'rb_xnli_en_formal_primary'}[task]
  return canonical_mc(folder,name,'XNLI' in task)
 if stage in ['mmlu','gmmlu']:return canonical_mc(folder,'mmlu_*' if stage=='mmlu' else 'global_mmlu_full_sw_*')
 if stage=='core':return audit.core(folder,TASK_MAP[task])
 if stage=='summary':return audit.summary(folder/'generation',folder/'scores','sw')
 if stage=='code':return audit.code(folder)
 raise AssertionError(stage)

def locate(profile,task):
 stage='reading' if task in ['SW Belebele','EN Belebele','SW XNLI','EN XNLI'] else {'SW XL-Sum':'summary','SW Global-MMLU':'gmmlu','EN MMLU':'mmlu','GSM8K flexible':'core','IFEval prompt strict':'core','HumanEval+':'code'}.get(task)
 folder=L/profile/(stage or '')
 if stage and (folder/'DONE.json').exists():return stage_reader(folder,stage,task)
 if profile in ['matched_fft','fft_full36']:return None
 if profile=='Base':
  value=audit.locate('sw','Base',TASK_MAP[task])
  if 'XNLI' in task:
   return canonical_mc(RB/'formal-base-xnli/results','rb_xnli_'+('sw' if task.startswith('SW') else 'en')+'_formal_primary',True)
  return value
 if profile in ['historical_fft','historical_ssu']:
  role='fft' if profile=='historical_fft' else 'ssu'
  if task=='HumanEval+':return None
  if stage=='reading':
   if 'XNLI' in task:return canonical_mc(OLD/'supplemental-benchmarks-v1'/role/'results','aya4b_xnli_'+('sw' if task.startswith('SW') else 'en'),True)
   return canonical_mc(OLD/(role+'-harness-v1/belebele/results'),'belebele_swh_Latn_locked' if task.startswith('SW') else 'belebele_eng_Latn_locked')
  if stage in ['mmlu','gmmlu']:return canonical_mc(OLD/(role+'-harness-v1')/stage/'results','mmlu_*' if stage=='mmlu' else 'global_mmlu_full_sw_*')
  if stage=='core':return audit.core(OLD/(role+'-generation-v2/core/results'),TASK_MAP[task])
  if stage=='summary':return audit.summary(OLD/(role+'-generation-v2/xlsum_sw_v2'),OLD/(role+'-generation-v2/xlsum_sw_v2-scores'),'sw')
 name={'p2_all':'all_layer','p2_full36':'full_feature','p2_ce':'ce_replay'}[profile]
 if stage=='reading':
  if 'XNLI' in task:return canonical_mc(RB/('formal-'+name+'-xnli/results'),'rb_xnli_'+('sw' if task.startswith('SW') else 'en')+'_formal_primary',True)
  return canonical_mc(P2/name/'hf/results','belebele_swh_Latn_locked' if task.startswith('SW') else 'belebele_eng_Latn_locked')
 if stage=='gmmlu':return canonical_mc(P2/name/'gmmlu/results','global_mmlu_full_sw_*')
 if stage=='mmlu':
  folder=audit.L/'sw_all/mmlu'
  return canonical_mc(folder,'mmlu_*') if profile=='p2_all' and (folder/'DONE.json').exists() else None
 if stage=='core':return audit.core(P2/name/'core/results',TASK_MAP[task])
 if stage=='code':return audit.code(RB/('formal-'+name+'-code'))
 if stage=='summary':return audit.summary(P2/name/'summary',P2/name/'summary-scores','sw')
 raise AssertionError((profile,task))

def translation(profile,task):
 direction='en_sw' if task.startswith('EN→') else 'sw_en';stage='mt_'+direction
 folder=L/profile/stage;dataset='flores200_'+direction+'_full_devtest_v2'
 if (folder/'DONE.json').exists():run=folder/'generation';scored=folder/'scores'
 elif profile in ['Base','historical_fft','historical_ssu']:
  role={'Base':'base','historical_fft':'fft','historical_ssu':'ssu'}[profile];run=OLD/(role+'-generation-v2')/dataset;scored=run.with_name(dataset+'-scores')
 else:return None
 manifest=audit.load(run/'manifest.json');scorefile=scored/'summary.json';aggregate=audit.load(scorefile)
 assert aggregate['samples']==1012 and audit.sha(run/'predictions.jsonl')==aggregate['predictions_sha256']
 xs=list(audit.rows(run/'predictions.jsonl'));assert len(xs)==1012
 assert [x['id'] for x in xs]==manifest['sample_ids']
 assert len({x['id'] for x in xs})==1012
 return xs,[dict(path=str(scorefile),sha256=audit.sha(scorefile),samples=1012)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
 from sacrebleu.metrics import CHRF
 import importlib.metadata
 assert importlib.metadata.version('sacrebleu')=='2.6.0'
 reported={(x['profile'],x['task']):float(x['score']) for x in csv.DictReader((Q/'scores.csv').open()) if x['score']!=''}
 vectors={};mts={};provenance=[];absolute=[];missing=[];numeric=Q/'numeric';numeric.mkdir(exist_ok=True)
 for profile in PROFILES:
  for task in TASKS:
   key=(profile,task)
   if key not in reported:missing.append(key);continue
   mt='chrF++' in task;found=translation(profile,task) if mt else locate(profile,task)
   assert found is not None,('Aggregate has no audited raw source',key)
   value,files=found
   if mt:
    scorer=CHRF(char_order=6,word_order=2,beta=2,lowercase=False,whitespace=False,eps_smoothing=False)
    stats=np.asarray(scorer._extract_corpus_statistics([x['prediction'] for x in value],[[x['reference'] for x in value]]),dtype=np.int64)
    score=scorer._compute_score_from_stats(stats.sum(0).tolist()).score
    identities=[audit.digest([x['id'],x['source'],x['reference'],x['prompt_token_sha256']]) for x in value]
    mts[key]=dict(stats=stats,identities=identities,scorer=scorer)
    (numeric/(profile+'_'+('en_sw' if task.startswith('EN→') else 'sw_en')+'_chrf.json')).write_text(json.dumps(dict(profile=profile,task=task,statistics=stats.tolist(),item_id_sha256=[audit.digest(x['id']) for x in value],ordered_input_identity_sha256=audit.digest(identities),sacrebleu='2.6.0',score=score,no_raw_text=True),separators=(',',':'))+'\n')
    n=len(value)
   else:
    vectors[key]=value;score=float(np.mean([x['value'] for x in value.values()]));n=len(value)
   assert abs(score-reported[key])<1e-7,(key,score,reported[key])
   absolute.append(dict(profile=profile,task=task,score=score,samples=n));provenance.append(dict(profile=profile,task=task,files=files))
 contrasts=[('Base','historical_fft'),('Base','matched_fft'),('Base','fft_full36'),('historical_fft','matched_fft'),('matched_fft','fft_full36'),('p2_all','p2_full36'),('p2_full36','fft_full36'),('p2_ce','fft_full36'),('historical_ssu','fft_full36')]
 intervals=[]
 for ci,(a,b) in enumerate(contrasts):
  for ti,task in enumerate(TASKS):
   ka=(a,task);kb=(b,task)
   if ka in vectors and kb in vectors:
    intervals.append(dict(reference=a,comparison=b,task=task,**audit.paired(vectors[ka],vectors[kb],np.random.SeedSequence([42,ci,ti]))))
   elif ka in mts and kb in mts:
    va,vb=mts[ka],mts[kb];assert va['identities']==vb['identities'],'Translation effective prompts/references changed'
    scorer=va['scorer'];rng=np.random.default_rng(np.random.SeedSequence([42,ci,ti]));deltas=[]
    for _ in range(2000):
     idx=rng.integers(1012,size=1012);deltas.append(scorer._compute_score_from_stats(vb['stats'][idx].sum(0).tolist()).score-scorer._compute_score_from_stats(va['stats'][idx].sum(0).tolist()).score)
    lo,hi=np.quantile(deltas,[.025,.975]);intervals.append(dict(reference=a,comparison=b,task=task,delta=reported[kb]-reported[ka],ci95_low=float(lo),ci95_high=float(hi),samples=1012,clusters=1012,resamples=2000,task_identity_sha256=audit.digest(va['identities'])))
 for task in TASKS:
  models=[m for m in PROFILES if (m,task) in vectors]
  if not models:continue
  anchor=vectors[models[0],task];keys=sorted(anchor,key=str)
  for m in models:
   v=vectors[m,task];assert v.keys()==anchor.keys() and all(v[k]['identity']==anchor[k]['identity'] and v[k]['cluster']==anchor[k]['cluster'] for k in keys),(m,task,'Input mismatch')
  _,index=np.unique([str(anchor[k]['cluster']) for k in keys],return_inverse=True)
  (numeric/(TASK_MAP[task]+'.json')).write_text(json.dumps(dict(task=task,models=models,item_id_sha256=[audit.digest(str(k)) for k in keys],cluster_index=index.tolist(),values=[[vectors[m,task][k]['value'] for m in models] for k in keys],ordered_input_identity_sha256=audit.digest([(str(k),anchor[k]['identity']) for k in keys]),no_raw_text=True),separators=(',',':'))+'\n')
 for filename,data in [('audited_scores.csv',absolute),('paired_changes.csv',intervals)]:
  if data:
   with (Q/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 result=dict(status='INCOMPLETE' if missing else 'ALL_96_CELLS_RAW_AUDITED',verified_cells=len(absolute),required_cells=96,missing=missing,paired_contrasts=len(intervals),bootstrap_resamples=2000,note='Paired benchmark uncertainty for fixed models, not training-seed robustness; unadjusted comparisons; no method selection. Translation resamples corpus statistics, never averages sentence chrF.',xnli_identity='Exact effective prompt hash, target hash, raw ID/prompt/label, and premise cluster; inert added metadata excluded')
 (Q/'score_audit.json').write_text(json.dumps(result,indent=2)+'\n');(Q/'score_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='missing'}))
 if not args.partial:assert not missing,missing
if __name__=='__main__':main()
