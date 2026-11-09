"""Deterministic build-only calibration; reference-answer CE, never formal tests."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import hashlib,json,tarfile
import pyarrow.parquet as pq
from transformers import AutoTokenizer
R=Path('/path/to/project')
O=Path('/path/to/storage/datasets/function-aware-formal-20261109/calibration-v1')
B=Path('/path/to/storage/datasets/aya-4b-multilang-20261106')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def priority(x):return hashlib.sha256(('fa-frozen-calibration-v1/'+x).encode()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=False)
 tok=AutoTokenizer.from_pretrained('/path/to/models/Qwen3-4B',local_files_only=True)
 old=Path('/path/to/storage/datasets/functional-representation-20261109/p2-v1/build_pool.jsonl')
 rowsold=[dict(x,task='old_'+x['capability']) for x in map(json.loads,old.open())]
 swdev=Path('/path/to/storage/datasets/functional-risk-p3-20261109/v3/development.jsonl')
 sw=[x for x in map(json.loads,swdev.open()) if x['split']=='build' and x['task'] in ['sw_reading','sw_summary']]
 assert len(sw)==96
 # CE reading only: the obsolete v3 MC distractors are never scored or used here.
 sw=[{k:v for k,v in x.items() if k in ['id','source_id','split','task','input_ids','prefix_length']} for x in sw]
 (O/'sw.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rowsold+sw))
 qa=pq.read_table(B/'raw/tydiqa-secondary-train.parquet').to_pylist()
 usedsources={}
 for lang,native in [('ar','arabic'),('ru','russian')]:
  prov=B/f'mixtures/{lang}/train6400.provenance.jsonl';used={x['source_id'] for x in map(json.loads,prov.open())}
  usedcontexts={x['context'] for x in qa if x['id'] in used};seen=set();target=[]
  def add(identifier,task,prompt,answer):
   prefix=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
   full=tok.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':answer}],tokenize=True,enable_thinking=False)
   if len(full)>2048 or len(prefix)>=len(full):return False
   assert full[:len(prefix)]==prefix
   target.append(dict(id=identifier,task=task,split='build',input_ids=full,prefix_length=len(prefix)));return True
  for x in sorted((x for x in qa if x['id'].startswith(native+'-')),key=lambda x:priority(x['id'])):
   if x['id'] in used or x['context'] in usedcontexts or x['context'] in seen:continue
   answer=x['answers']['text'][0];start=x['answers']['answer_start'][0]
   if x['context'][start:start+len(answer)]!=answer:continue
   prompt=('اقرأ المقطع التالي ثم أجب عن السؤال باللغة العربية. قدم إجابة قصيرة تستند إلى المقطع فقط.\n\nالمقطع:\n'+x['context']+'\n\nالسؤال: '+x['question']) if lang=='ar' else ('Прочитайте следующий отрывок и ответьте на вопрос по-русски. Дайте краткий ответ, используя только сведения из отрывка.\n\nОтрывок:\n'+x['context']+'\n\nВопрос: '+x['question'])
   if add('tydi-'+x['id'],lang+'_reading',prompt,answer):seen.add(x['context'])
   if len(target)==64:break
  assert len(target)==64,(lang,len(target))
  arc=B/'raw'/(native+'_XLSum_v2.0.tar.bz2')
  with tarfile.open(arc) as t:
   member=next(m for m in t if Path(m.name).name==native+'_val.jsonl')
   summaries=[json.loads(s) for s in t.extractfile(member)]
  for x in sorted(summaries,key=lambda x:priority(x['id'])):
   if x['id'] in used:continue
   prefix='لخص المقال التالي باللغة العربية. حافظ على الحقائق الرئيسية وقدم ملخصًا موجزًا فقط.\n\n' if lang=='ar' else 'Кратко изложите следующую статью по-русски. Сохраните основные факты и выведите только краткое резюме.\n\n'
   if len(tok.encode(x['summary']))>240:continue
   add('xlsum-val-'+x['id'],lang+'_summary',prefix+x['text'],x['summary'])
   if len(target)==96:break
  assert len(target)==96
  (O/(lang+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rowsold+target))
  usedsources[lang]={str(p):sha(p) for p in [prov,arc]}
 manifest=dict(status='FROZEN_BEFORE_NEW_CALIBRATION_INFERENCE',old_build128_reused=True,per_language=dict(old_math=64,old_instruction=64,target_reading=64,target_summary=32),target_scope='unused TyDiQA train articles and official XL-Sum validation; SW reuses previous P3 build CE; not target benchmark effectiveness evidence',files={p.name:sha(p) for p in O.glob('*.jsonl')},sources=usedsources,old_pool_sha256=sha(old),sw_development_sha256=sha(swdev),all_formal_tests_used=False)
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
