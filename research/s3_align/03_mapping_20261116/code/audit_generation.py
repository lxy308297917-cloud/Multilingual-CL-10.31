"""Actual stopping/token accounting and extraction artifacts; no main rescoring."""
from common import *
from jsonl_safe import rows
from math_error_audit import number
import argparse,collections,csv
from transformers import AutoTokenizer
def object_sha(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['math256','formal'],required=True);a=ap.parse_args()
 models=['Base','A','B','E'] if a.scope=='math256' else ['Base','T0','T1','T2'];tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True);summary=[];details=[];sources={}
 for model in models:
  root=LOG/'math256'/model if a.scope=='math256' else LOG/'evaluation'/model/'core'
  assert read(root/'DONE.json')['returncode']==0
  trace=rows(root/'completion-metadata.jsonl');lookup={(r['prompt_token_sha256'],r['response_sha256']):r for r in trace};assert len(lookup)==len(trace)==(256 if a.scope=='math256' else 1860)
  sources[str(root/'completion-metadata.jsonl')]=sha(root/'completion-metadata.jsonl');used=set()
  tasks=[('GSM8K','samples_s3map_math256_*.jsonl',256)] if a.scope=='math256' else [('GSM8K','samples_sft_v1_gsm8k_*.jsonl',1319),('IFEval','samples_sft_v1_ifeval_*.jsonl',541)]
  for task,pattern,n in tasks:
   files=list((root/'results').rglob(pattern));assert len(files)==1;sources[str(files[0])]=sha(files[0]);rr=rows(files[0]);rr=[r for r in rr if r['filter']==('flexible-extract' if task=='GSM8K' else 'none')];assert len(rr)==n
   records=[]
   for r in rr:
    prompt=r['arguments']['gen_args_0']['arg_0'];text=r['resps'][0][0];key=(object_sha(tok.encode(prompt,add_special_tokens=False)),object_sha(text));assert key in lookup
    generation=lookup[key];used.add(key);is_numeric_format=False
    if task=='GSM8K':
     prediction=number(r['filtered_resps'][0]);gold=number(r['doc']['answer'].split('####')[-1]);is_numeric_format=not r['exact_match'] and prediction is not None and gold is not None and prediction==gold
    lines=[s.strip() for s in text.splitlines() if len(s.strip())>=20];repeated=sum(v-1 for v in collections.Counter(lines).values() if v>1)
    item=dict(model=model,task=task,doc_id=r['doc_id'],generated_tokens=generation['generated_tokens'],finish_reason=generation['finish_reason'],empty=not text.strip(),missing_final_marker=('####' not in text) if task=='GSM8K' else None,numeric_format_false_negative=bool(is_numeric_format),repeated_exact_lines=repeated,primary_score=r['exact_match'] if task=='GSM8K' else r['prompt_level_strict_acc'])
    records.append(item);details.append(item)
   summary.append(dict(model=model,task=task,n=n,mean_generated_tokens=sum(r['generated_tokens'] for r in records)/n,length_limit=sum(r['finish_reason']=='length' for r in records),empty=sum(r['empty'] for r in records),missing_final_marker=sum(bool(r['missing_final_marker']) for r in records) if task=='GSM8K' else None,numeric_format_false_negatives=sum(r['numeric_format_false_negative'] for r in records),outputs_with_repeated_exact_lines=sum(r['repeated_exact_lines']>0 for r in records)))
  assert len(used)==len(trace)
 for name,rs in [('summary',summary),('items',details)]:
  with (R/'analysis'/f'{a.scope}_generation_{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 write(R/'records'/f'{a.scope.upper()}_GENERATION_AUDIT.json',dict(at=now(),scope=a.scope,sources=sources,join='Exact prompt-token hash and rawresponse hash',primary_scores_unchanged=True,repetition='Exact repeated nonemptylines>=20characters only, a heuristic,not broad semantic repetition',numeric_format='Already-extracted scalar numerically equals reference but officialstringmetricis0; doesnotrescoreorrepairoutputs'))
 print(json.dumps(summary))
if __name__=='__main__':main()
