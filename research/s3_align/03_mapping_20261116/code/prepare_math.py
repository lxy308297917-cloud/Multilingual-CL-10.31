from common import *
import re,random,pyarrow.parquet as pq,pyarrow as pa,yaml
from transformers import AutoTokenizer
def norm(s):return ' '.join(s.lower().split())
def main():
 assert not (R/'inputs/math256.jsonl').exists()
 source=Path('/srv/storage/datasets/benchmarks-20261031/gsm8k/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet')
 data=pq.read_table(source).to_pylist();excluded=set();evidence={}
 roots=[Path('/srv/storage/datasets/functional-representation-20261109'),Path('/srv/storage/datasets/functional-risk-p3-20261109'),Path('/srv/storage/datasets/function-aware-formal-20261109')]
 # Broad ID audit, including check/development pools and previous direction construction.
 for root in roots:
  for p in sorted(root.rglob('*.json*')):
   if p.stat().st_size>100_000_000:continue
   text=p.read_text();ids=set(map(int,re.findall(r'math-train-(\d+)',text)))
   if ids:excluded|=ids;evidence[str(p)]=dict(sha256=sha(p),ids=sorted(ids))
 # Also exclude the original Base screening candidate pool, not only its selected128.
 screen=HOME/'logs/qwen3-multilingual/functional-representation-20261109/base-screen-v1'
 for p in screen.rglob('*.jsonl'):
  text=p.read_text();ids=set(map(int,re.findall(r'math-train-(\d+)',text)))
  if ids:excluded|=ids;evidence[str(p)]=dict(sha256=sha(p),ids=sorted(ids))
 hist=read(R/'records/HISTORICAL_MATH_DONE.json')['sources']['Base']['files'][0]['path']
 prompt=rows(hist)[0]['arguments']['gen_args_0']['arg_0']
 fewshot=[i for i,r in enumerate(data) if r['question'] in prompt];excluded.update(fewshot)
 # Exact normalized question overlap against both English and Swahili main training prompts.
 texts={norm(r['instruction']) for lang in ['en','sw'] for r in rows(DATA/f'{lang}_train.jsonl')}
 overlap=[i for i,r in enumerate(data) if norm(r['question']) in texts];excluded.update(overlap)
 eligible=[i for i in range(len(data)) if i not in excluded];selected=random.Random(20261116).sample(eligible,256)
 tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True);result=[]
 for i in selected:
  d=data[i];user='Question: '+d['question']+'\nAnswer:'
  prefix=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  text=prefix+d['answer'];ids=tok.encode(text,add_special_tokens=False);prefix_ids=tok.encode(prefix,add_special_tokens=False)
  # Token boundaries at the prefix/answer join can merge; offsets define answer targets exactly.
  encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
  targets=[j for j,(s,e) in enumerate(encoded['offset_mapping']) if s>=len(prefix) and e>s]
  positions=sorted({targets[round((len(targets)-1)*k/3)]-1 for k in range(4)})
  assert min(positions)>=len(prefix_ids)-2 and len(ids)<2048
  result.append(dict(id=f'math-train-{i}',source_id=i,question=d['question'],answer=d['answer'],reference_final=d['answer'].split('####')[-1].strip(),split='new_training_side_diagnostic',input_ids=ids,prefix=prefix,positions=positions,target_ids=[ids[p+1] for p in positions],prompt_tokens=len(prefix_ids),answer_tokens=len(targets),reference_steps=len(d['answer'].splitlines())))
 jsonl(R/'inputs/math256.jsonl',result)
 pq.write_table(pa.Table.from_pylist([data[i] for i in selected]),R/'inputs/math256.parquet')
 task=yaml.safe_load((P/'configs/sft_v1/eval_tasks/gsm8k.yaml').read_text());task['task']='s3map_math256';task['dataset_kwargs']['data_files']['test']=str(R/'inputs/math256.parquet')
 (R/'inputs/math_task').mkdir();(R/'inputs/math_task/math256.yaml').write_text(yaml.safe_dump(task,sort_keys=False))
 write(R/'records/MATH_DATA_FROZEN.json',dict(at=now(),source=str(source),source_sha256=sha(source),source_rows=len(data),excluded_ids=sorted(excluded),excluded_fewshot_ids=fewshot,excluded_exact_sft_overlap=overlap,exclusion_evidence=evidence,selected_ids=selected,seed=20261116,selection='Uniform source-ID sample before any new model predictions; no Base-correct selection',models=['Base','A','B','E'],layers=[12,18,24],teacher_forcing='Zero-shot identical reference prefix,4 uniformly spaced reference-answer token prediction positions; separate from5-shot actual-generation benchmark protocol',generation='Frozen upstream GSM8K task,5shot greedy,max1280,flexible-extract primary; training-source test split explicitly exploratory',limitation='No claim to exhaustively know every historical training exposure; exclusions cover known local functional pools,screening candidates,exact SFT prompt overlap and actual5shot examples.',hashes={str(p):sha(p) for p in [R/'inputs/math256.jsonl',R/'inputs/math256.parquet',R/'inputs/math_task/math256.yaml']}))
 print('math frozen',len(result),'excluded',len(excluded),'eligible',len(eligible))
if __name__=='__main__':main()
