"""Offline numeric verification. Does not load models, execute candidates or need test text."""
import argparse,collections,csv,gzip,hashlib,json,math,statistics
from pathlib import Path

def rows(p):
 op=gzip.open if p.suffix=='.gz' else open
 with op(p,'rt') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-9):assert math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol),(a,b)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('research_directory',type=Path);a=ap.parse_args();root=a.research_directory
 def get(path):
  p=root/path
  return rows(p if p.exists() else Path(str(p)+'.gz'))
 scores=get('benchmarks/new_scores.csv');xnli=get('benchmarks/xnli_items.csv');code=get('benchmarks/code_items.csv')
 groups=collections.defaultdict(list)
 for x in xnli:
  ll=[float(x['loglikelihood_'+l]) for l in 'ABC'];p=max(range(3),key=lambda i:ll[i]);assert p==int(x['predicted_letter']);assert int(x['correct'])==int(p==int(x['gold_letter']))
  assert int(x['correct'])==int(x['gold_semantic']==x['predicted_semantic'])
  close(x['correct_answer_margin'],ll[int(x['gold_letter'])]-max(v for i,v in enumerate(ll) if i!=int(x['gold_letter'])))
  groups[x['model'],x['language'],x['protocol']].append(x)
 assert len(xnli)==80160 and len(groups)==16 and all(len(xs)==5010 for xs in groups.values())
 for s in scores:
  if s['task'].startswith('XNLI '):
   xs=groups[s['model'],s['task'][-2:].lower(),s['protocol']];close(s['score'],sum(int(x['correct']) for x in xs)/5010*100)
 for m in ['Base','all_layer','full_feature','ce_replay']:
  xs=[x for x in code if x['model']==m];assert len(xs)==164 and len({x['id'] for x in xs})==164
  for x in xs:
   assert int(x['base_pass'])==int(x['base_status']=='pass')
   assert int(x['plus_pass'])==int(x['base_status']=='pass' and x['plus_status']=='pass')
  for key,task in [('base','HumanEval'),('plus','HumanEval+')]:
   s=next(s for s in scores if s['model']==m and s['task']==task);close(s['score'],sum(int(x[key+'_pass']) for x in xs)/164*100)
 samples=root/'benchmarks/code_generation_samples.jsonl.gz'
 if samples.exists():
  identities={(x['model'],x['id']):x['solution_sha256'] for x in code}
  with gzip.open(samples,'rt') as f:
   generated=[json.loads(line) for line in f]
  assert len(generated)==656
  for x in generated:
   digest=hashlib.sha256(x['sanitized_solution'].encode()).hexdigest()
   assert digest==identities[x['model'],x['id']]==x['solution_sha256']
 drifts=get('mechanism/old_drift_items.csv');layer=get('mechanism/old_drift_layers.csv');g=collections.defaultdict(list)
 for x in drifts:g[x['model'],x['split'],x['capability'],x['layer']].append(float(x['normalized_squared_l2']))
 for x in layer:
  vs=g[x['model'],x['split'],x['capability'],x['layer']];assert len(vs)==64
  close(x['mean'],statistics.fmean(vs));close(x['median'],statistics.median(vs));close(x['std'],statistics.pstdev(vs))
  if x['model']=='Base':assert all(v==0 for v in vs)
 sim=get('mechanism/cross_language_items.csv');simlayers=get('mechanism/cross_language_layers.csv');g=collections.defaultdict(list)
 base_cos={(x['condition'],x['id'],x['layer']):float(x['cosine']) for x in sim if x['model']=='Base'}
 for x in sim:
  close(x['delta_vs_base'],float(x['cosine'])-base_cos[x['condition'],x['id'],x['layer']])
  g[x['model'],x['condition'],x['layer']].append(x)
 for x in simlayers:
  vs=g[x['model'],x['condition'],x['layer']];assert len(vs)==300
  for source,target in [('cosine','mean'),('delta_vs_base','delta_mean_vs_base'),('shuffled_cosine','shuffled_mean')]:close(x[target],statistics.fmean(float(v[source]) for v in vs))
  close(x['std'],statistics.pstdev(float(v['cosine']) for v in vs))
  close(x['matched_minus_shuffled_mean'],statistics.fmean(float(v['cosine'])-float(v['shuffled_cosine']) for v in vs))
 # Covariance summaries require raw hidden features for end-to-end re-extraction;
 # the separate metrics.py dense-vs-Gram oracle validates that computation.
 print(json.dumps(dict(status='PASS',xnli_likelihood_items=len(xnli),executed_code_items=len(code),drift_items=len(drifts),cosine_items=len(sim),covariance_end_to_end='Requires hashed private feature cache or re-extraction; not falsely claimed by this aggregate verifier')))
if __name__=='__main__':main()
