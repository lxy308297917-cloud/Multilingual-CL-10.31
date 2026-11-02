"""Deterministic output diagnostics; no model judging or generated-code execution."""
import argparse,json,math
from pathlib import Path
import numpy as np
from common import dump,jsonl,require_complete


def rank(values):
 values=np.asarray(values);order=np.argsort(values,kind='stable');ranks=np.empty(len(values),float);i=0
 while i<len(values):
  j=i+1
  while j<len(values) and values[order[j]]==values[order[i]]:j+=1
  ranks[order[i:j]]=(i+j-1)/2;i=j
 return ranks


def spearman(a,b):
 if len(a)<3:return None
 x,y=rank(a),rank(b)
 if np.std(x)==0 or np.std(y)==0:return None
 return float(np.corrcoef(x,y)[0,1])


def main():
 p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 state=json.loads(a.state.read_text());records=[];summaries={}
 for key,info in state['states'].items():
  if info['status']!='COMPLETE' or not key.endswith('|full'):continue
  method,task,_=key.split('|');run=Path(info['output']);m=json.loads((run/'manifest.json').read_text());rows=require_complete(jsonl(run/'predictions.jsonl'),m['sample_ids']);s=json.loads(Path(info['summary']).read_text());n=len(rows)
  item=dict(method=method,benchmark=task,samples=n,generated_tokens_mean=float(np.mean([x['generated_tokens'] for x in rows])),generated_tokens_p50=float(np.median([x['generated_tokens'] for x in rows])),generated_tokens_p95=float(np.percentile([x['generated_tokens'] for x in rows],95)),input_tokens_mean=float(np.mean([x['input_tokens'] for x in rows])),cap_count=sum(x['generation_hit_cap'] for x in rows),source_exact_copy_count=sum(x['prediction'].strip()==str(x.get('source','')).strip() for x in rows))
  diagpath=Path(info['summary']).parent/'diagnostics_per_sample.json'
  if diagpath.exists():
   diag=json.loads(diagpath.read_text());measured=[x for x in diag if x['language_match'] is not None];item.update(empty_count=sum(x['empty'] for x in diag),repetition_count=sum(x['repetitive'] for x in diag),language_measured=len(measured),language_mismatch_count=sum(not x['language_match'] for x in measured),language_mismatch_rate_among_measured=sum(not x['language_match'] for x in measured)/len(measured) if measured else None,language_unknown=n-len(measured))
  if task.startswith('xlsum'):
   rouge=json.loads((Path(info['summary']).parent/'rouge_per_sample.json').read_text());scores={x['id']:x['ROUGE-Lsum'] for x in rouge};lengths=[x['article_tokens_before_truncation'] for x in rows];values=[scores[x['id']] for x in rows];cuts=np.percentile(lengths,[25,50,75]);bands=[]
   for band in range(4):
    inds=[i for i,x in enumerate(lengths) if int(np.searchsorted(cuts,x,side='right'))==band];bands.append(dict(quartile=band+1,samples=len(inds),rouge_lsum=float(np.mean([values[i] for i in inds])) if inds else None))
   item.update(article_tokens_quantiles=dict(zip(['min','p25','p50','p75','p95','max'],[float(x) for x in np.percentile(lengths,[0,25,50,75,95,100])])),article_truncated_count=sum(x['article_truncated'] for x in rows),length_rouge_lsum_spearman=spearman(lengths,values),length_quartile_cutpoints=cuts.tolist(),length_quartile_scores=bands,source_copy_scope='Exact whole-article copy only; does not measure partial extractiveness')
   summaries[(task,method)]=(rows,scores)
  if task.startswith(('human','mbpp')):item.update(static_diagnostics=s['static_diagnostics'],execution_status='NOT_RUN')
  records.append(item)
 for item in records:
  key=(item['benchmark'],item['method']);basekey=(item['benchmark'],'base')
  if key not in summaries or basekey not in summaries or item['method']=='base':continue
  rows,scores=summaries[key];br,base=summaries[basekey]
  if [x['id'] for x in rows]!=[x['id'] for x in br]:raise ValueError('Length comparison IDs differ')
  item['article_length_paired_rouge_delta_spearman']=spearman([x['article_tokens_before_truncation'] for x in rows],[scores[x['id']]-base[x['id']] for x in rows])
 a.output.mkdir(parents=True);dump(a.output/'output_diagnostics.json',dict(results=records,limitations=['Language detection is an auxiliary classifier with unmeasured short outputs, not ground truth translation-direction accuracy.','Length correlations and quartiles are descriptive, not causal.','Exact source-copy is stricter than partial extractiveness; paraphrase and factuality are unmeasured.','Generated-code syntax/signature heuristics cannot establish test pass rates.']))
 print(json.dumps({'diagnosed_full_runs':len(records),'output':str(a.output)}))
if __name__=='__main__':main()
