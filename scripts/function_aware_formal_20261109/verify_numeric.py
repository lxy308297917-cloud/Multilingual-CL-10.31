"""Offline recomputation of exported means, corpus metrics and paired intervals.

Requires NumPy and SacreBLEU2.6.0. Input is the published research directory;
validation scope is numeric consistency, with source/model identity supported
separately by the server-side raw audits.
"""
from pathlib import Path
import argparse,csv,json,importlib.metadata
import numpy as np
FA_TASKS=['target_reading','target_summary','target_knowledge','target_xnli','en_gsm8k','en_ifeval','en_mmlu','en_code','en_reading','en_xnli']
FFT_TASKS=['SW Belebele','SW XL-Sum','SW Global-MMLU','SW XNLI','EN Belebele','EN XNLI','EN MMLU','GSM8K flexible','IFEval prompt strict','HumanEval+','EN→SW chrF++','SW→EN chrF++']
FA_AUX=['GSM8K strict','IFEval instruction strict'];FFT_AUX=FA_AUX+['EN→SW BLEU','SW→EN BLEU']
FA_CONTRASTS=[('Base','all'),('Base','full'),('Base','kd'),('Base','random'),('Base','function'),('all','full'),('all','kd'),('all','random'),('all','function'),('full','function'),('random','function'),('kd','function'),('HistoricalFull36','function')]
FFT_CONTRASTS=[('Base','historical_fft'),('Base','matched_fft'),('Base','fft_full36'),('historical_fft','matched_fft'),('matched_fft','fft_full36'),('p2_all','p2_full36'),('p2_full36','fft_full36'),('p2_ce','fft_full36'),('historical_ssu','fft_full36')]
def load(p):return json.loads(p.read_text())
def close(a,b):assert abs(float(a)-float(b))<1e-7,(a,b)
def corpus_scorer(task):
 assert importlib.metadata.version('sacrebleu')=='2.6.0'
 from sacrebleu.metrics import BLEU,CHRF
 return BLEU(lowercase=False,tokenize='13a',smooth_method='exp',smooth_value=None,effective_order=False) if task.endswith('BLEU') else CHRF(char_order=6,word_order=2,beta=2,lowercase=False,whitespace=False,eps_smoothing=False)
def point(v):
 if v['kind']=='mean':return float(v['values'].mean())
 if v['kind']=='ratio':return float(100*v['values'].sum()/v['denominator'].sum())
 return v['scorer']._compute_score_from_stats(v['values'].sum(0).tolist()).score

def parse_vectors(root):
 vectors={}
 for phase,folder in [('primary','numeric'),('auxiliary','auxiliary_numeric')]:
  for file in sorted((root/folder).glob('*.json')):
   d=load(file);lang=d.get('language','');identity=d['ordered_input_identity_sha256'];ids=d['item_id_sha256'];n=len(ids)
   assert n and len(set(ids))==n and all(len(x)==64 for x in ids) and len(identity)==64
   if 'statistics' in d:
    array=np.asarray(d['statistics'],dtype=np.int64);assert len(array)==n and np.all(array>=0)
    entries=[(d['profile'],dict(kind='corpus',values=array,scorer=corpus_scorer(d['task'])))]
   elif 'values' in d:
    array=np.asarray(d['values'],dtype=float);clusters=np.asarray(d['cluster_index'],dtype=int)
    assert array.shape==(n,len(d['models'])) and len(clusters)==n and np.array_equal(np.unique(clusters),np.arange(clusters.max()+1)) and np.isfinite(array).all() and (array>=0).all() and (array<=100).all()
    entries=[(m,dict(kind='mean',values=array[:,i],clusters=clusters)) for i,m in enumerate(d['models'])]
   elif 'numerators' in d:
    array=np.asarray(d['numerators'],dtype=float);den=np.asarray(d['denominators'],dtype=float)
    assert array.shape==(n,len(d['models'])) and den.shape==(n,) and (den>0).all() and (array>=0).all() and (array<=den[:,None]).all()
    entries=[(m,dict(kind='ratio',values=array[:,i],denominator=den)) for i,m in enumerate(d['models'])]
   else:
    array=np.asarray(d['rows'],dtype=float);assert array.shape==(n,2) and (array[:,1]>0).all() and (array[:,0]>=0).all() and (array[:,0]<=array[:,1]).all()
    entries=[(d['profile'],dict(kind='ratio',values=array[:,0],denominator=array[:,1]))]
   for model,value in entries:
    key=(phase,lang,model,d['task']);assert key not in vectors
    value.update(identity=identity,item_ids=ids);vectors[key]=value
 return vectors

def interval(a,b,seed):
 assert a['kind']==b['kind'] and a['identity']==b['identity'] and a['item_ids']==b['item_ids'],'Different paired inputs'
 rng=np.random.default_rng(np.random.SeedSequence(seed));kind=a['kind'];draws=[]
 if kind=='mean':
  assert np.array_equal(a['clusters'],b['clusters']);inverse=a['clusters'];n=int(inverse.max())+1
  sums=np.bincount(inverse,weights=b['values']-a['values'],minlength=n);sizes=np.bincount(inverse,minlength=n)
  for _ in range(40):
   idx=rng.integers(n,size=(50,n));draws.extend(sums[idx].sum(1)/sizes[idx].sum(1))
 elif kind=='ratio':
  assert np.array_equal(a['denominator'],b['denominator']);n=len(a['values']);dif=b['values']-a['values'];den=a['denominator']
  for _ in range(2000):
   idx=rng.integers(n,size=n);draws.append(100*dif[idx].sum()/den[idx].sum())
 else:
  n=len(a['values']);scorer=a['scorer']
  for _ in range(2000):
   idx=rng.integers(n,size=n);draws.append(scorer._compute_score_from_stats(b['values'][idx].sum(0).tolist()).score-scorer._compute_score_from_stats(a['values'][idx].sum(0).tolist()).score)
 low,high=np.quantile(draws,[.025,.975]);return point(b)-point(a),low,high

def main():
 ap=argparse.ArgumentParser();ap.add_argument('research_directory',type=Path);ap.add_argument('--family',choices=['fa','fft'],required=True);ap.add_argument('--require-complete',action='store_true');a=ap.parse_args();root=a.research_directory;v=parse_vectors(root);expected={}
 for phase,name in [('primary','audited_scores.csv'),('auxiliary','auxiliary_scores.csv')]:
  for row in csv.DictReader((root/name).open()):
   if row['score']=='':continue
   key=(phase,row.get('language',''),row['method'] if a.family=='fa' else row['profile'],row['task']);assert key not in expected;expected[key]=float(row['score'])
 assert v.keys()==expected.keys(),dict(missing=list(expected.keys()-v.keys()),extra=list(v.keys()-expected.keys()))
 for key,value in v.items():
  close(point(value),expected[key])
  if value['kind']=='ratio' and 'IFEval' in key[-1]:assert len(value['values'])==541 and value['denominator'].sum()==834
 checked=0
 for phase,name in [('primary','paired_changes.csv'),('auxiliary','auxiliary_paired_changes.csv')]:
  for row in csv.DictReader((root/name).open()):
   lang=row.get('language','');contrast=(row['reference'],row['comparison']);ci=(FA_CONTRASTS if a.family=='fa' else FFT_CONTRASTS).index(contrast)
   tasks=(FA_TASKS if a.family=='fa' else FFT_TASKS) if phase=='primary' else (FA_AUX if a.family=='fa' else FFT_AUX);ti=tasks.index(row['task'])
   seed=[42,['sw','ar','ru'].index(lang),ci,ti] if a.family=='fa' else [42,ci,ti]
   va,vb=v[phase,lang,row['reference'],row['task']],v[phase,lang,row['comparison'],row['task']]
   actual=interval(va,vb,seed)
   for x,key in zip(actual,['delta','ci95_low','ci95_high']):close(x,row[key])
   checked+=1
 primary=sum(k[0]=='primary' for k in v);auxiliary=len(v)-primary;required_primary=190 if a.family=='fa' else 96;required_auxiliary=38 if a.family=='fa' else 32
 complete=primary==required_primary and auxiliary==required_auxiliary
 if a.require_complete:assert complete,'Experiments are still incomplete'
 print(json.dumps(dict(status='PASS_AVAILABLE_NUMERIC_CONSISTENCY_SCOPE_ONLY',family=a.family,primary_score_cells=primary,auxiliary_score_cells=auxiliary,required_primary_score_cells=required_primary,required_auxiliary_score_cells=required_auxiliary,all_expected_cells_present=complete,paired_intervals_recomputed=checked,resamples=2000,numpy=np.__version__,sacrebleu=importlib.metadata.version('sacrebleu'),source_identity_scope='Original prompts, model files and code-execution identity are supported by the separate server-side audits; this command checks exported numerical consistency only.')))
if __name__=='__main__':main()
