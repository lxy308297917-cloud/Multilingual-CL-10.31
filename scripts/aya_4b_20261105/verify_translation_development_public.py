"""Recompute development corpus scores and paired intervals without raw text or GPU."""
import argparse,csv,gzip,json,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
from sacrebleu.metrics import CHRF,BLEU

def main():
 p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();assert importlib.metadata.version('sacrebleu')=='2.6.0'
 x=json.loads((a.analysis/'results.json').read_text());tables={};scores=0
 for spec in x['corpus_statistics']:
  f=a.analysis/spec['file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==spec['sha256']
  opener=gzip.open if f.suffix=='.gz' else open
  with opener(f,'rt',newline='') as stream:rows=list(csv.DictReader(stream))
  assert len(rows)==len({v['item_id_sha256'] for v in rows})==spec['samples']==64
  matrix=np.array([json.loads(v['statistics']) for v in rows],dtype=np.int64)
  scorer=CHRF(word_order=2) if spec['metric']=='chrF++' else BLEU(tokenize='13a')
  score=scorer._compute_score_from_stats(matrix.sum(axis=0).tolist()).score
  assert abs(score-spec['score'])<1e-10
  report=next(v for v in x['scores'] if all(v[k]==spec[k] for k in ['model','profile','direction']))
  assert abs(score-report[spec['metric']])<1e-10
  key=(spec['model'],spec['profile'],spec['direction'],spec['metric']);assert key not in tables;tables[key]=(rows,matrix,scorer);scores+=1
 checked=0
 for spec in x['paired_comparisons']:
  tr,ta,scorer=tables[tuple(spec['target'])+(spec['metric'],)];br,ba,_=tables[tuple(spec['base'])+(spec['metric'],)]
  assert [(r['item_id_sha256'],r['identity_sha256']) for r in tr]==[(r['item_id_sha256'],r['identity_sha256']) for r in br]
  compute=lambda z:scorer._compute_score_from_stats(z.sum(axis=0).tolist()).score
  delta=compute(ta)-compute(ba);rng=np.random.default_rng(spec['seed']);values=[]
  for _ in range(spec['repeats']):
   indices=rng.integers(0,len(ta),size=len(ta));values.append(compute(ta[indices])-compute(ba[indices]))
  lo,hi=np.percentile(values,[2.5,97.5]);assert max(abs(delta-spec['delta']),abs(lo-spec['ci95_low']),abs(hi-spec['ci95_high']))<1e-8;checked+=1
 assert scores==16 and checked==16
 result={'status':'VERIFIED_PUBLIC_DEVELOPMENT_SCORES_AND_INTERVALS','scores_recomputed':scores,'intervals_recomputed':checked,'statistics_files':len(tables),'samples_per_direction':64,'requires_raw_text_or_GPU':False,'formal_test':False};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
