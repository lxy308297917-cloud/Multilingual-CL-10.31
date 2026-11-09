"""Deterministic FLORES dev probe, excluding past dev64/demos and exact train text."""
import hashlib,json,re,unicodedata
from pathlib import Path
from transformers import AutoTokenizer
R=Path('/data0/users/lixiaoyang/10.31/multilingual-research')
D=Path('/data1/lixiaoyang/datasets/representation-benchmarks-20261109/v1')
F=Path('/data1/lixiaoyang/datasets/flores200-meta-20261031/flores200_dataset/dev')
OLD=Path('/data1/lixiaoyang/datasets/functional-representation-20261109/p1-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s)).strip().casefold()
def main():
 out=D/'parallel';out.mkdir(exist_ok=False)
 texts={k:(F/(v+'.dev')).read_text().splitlines() for k,v in [('en','eng_Latn'),('sw','swh_Latn')]};assert all(len(x)==997 for x in texts.values())
 shots={409,521,656};prev=set(sorted((i for i in range(997) if i not in shots),key=lambda i:hashlib.sha256(('aya4b-mt-decoding-dev-v1|'+str(i)).encode()).hexdigest())[:64]);exclude=shots|prev
 train=Path('/data1/lixiaoyang/datasets/aya-4b-20261105/mixture-v1/train6400.jsonl')
 pool=OLD/'encoded_selected.jsonl'
 tok=AutoTokenizer.from_pretrained('/data0/users/lixiaoyang/models/Qwen3-4B',local_files_only=True)
 # Exact normalized text containment only, not a claim of semantic decontamination.
 strings=[]
 def collect(x):
  if isinstance(x,str):strings.append(norm(x))
  elif isinstance(x,list):
   for v in x:collect(v)
  elif isinstance(x,dict):
   for v in x.values():collect(v)
 for line in train.open():collect(json.loads(line))
 assert pool.exists()
 for line in pool.open():collect(tok.decode(json.loads(line)['input_ids'],skip_special_tokens=True))
 normalized={lang:[norm(t) for t in texts[lang]] for lang in ['en','sw']}
 overlap={i for i in range(997) if any(any(normalized[lang][i] in s for s in strings) for lang in ['en','sw'])}
 eligible=[i for i in range(997) if i not in exclude|overlap and all(texts[l][i].strip() for l in ['en','sw'])]
 chosen=sorted(eligible,key=lambda i:hashlib.sha256(('representation-benchmarks-20261109-parallel-v1|'+str(i)).encode()).hexdigest())[:300];assert len(chosen)==300
 tok=AutoTokenizer.from_pretrained('/data0/users/lixiaoyang/models/Qwen3-4B',local_files_only=True)
 rows=[];encoded=[]
 for i in chosen:
  row=dict(id='flores-dev-'+str(i),source_index_zero_based=i,dataset='FLORES200',split='dev',en=texts['en'][i],sw=texts['sw'][i],quality='Official parallel dev; no empty strings; exact normalized train/old-pool overlap excluded; not native-speaker-certified')
  e=dict(id=row['id'],languages={})
  for lang in ['en','sw']:
   text=row[lang];ids=tok.encode(text,add_special_tokens=False)
   prompt=tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   assert prompt.count(text)==1
   begin=prompt.index(text);stop=begin+len(text)
   t=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
   assert t['input_ids']==tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False)
   positions=[j for j,(start,end) in enumerate(t['offset_mapping']) if end>start and start>=begin and end<=stop]
   assert positions and len(ids)<=2048 and len(t['input_ids'])<=2048
   e['languages'][lang]=dict(raw_ids=ids,chat_ids=t['input_ids'],chat_content_positions=positions)
   row[lang+'_tokens_raw']=len(ids);row[lang+'_tokens_chat']=len(t['input_ids']);row[lang+'_tokens_chat_content']=len(positions)
  rows.append(row);encoded.append(e)
 for name,xs in [('pairs.jsonl',rows),('encoded.jsonl',encoded)]:
  (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in xs))
 review=sorted(rows,key=lambda x:hashlib.sha256(('basic-review|'+x['id']).encode()).hexdigest())[:24]
 (out/'review24.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in review))
 report=dict(status='FROZEN_BEFORE_REPRESENTATION_EXTRACTION',rows=300,split='dev NOT devtest',selection='Fixed hash order; excludes prior64 development IDs,3 demos and exact normalized target-train/old-pool text containment',past64_excluded=sorted(prev),demonstration_ids=sorted(shots),exact_overlap_ids=sorted(overlap),eligible=len(eligible),source_hashes={p.name:sha(p) for p in [F/'eng_Latn.dev',F/'swh_Latn.dev',train,pool]},files={p.name:sha(p) for p in out.glob('*.jsonl')},old_encoded_sha256=sha(OLD/'encoded_selected.jsonl'),conditions=['raw_content_mean','chat_all_tokens_mean','chat_content_tokens_mean'],pooling='Each example separately, unpadded tokens only; embedding0 and36 raw transformer block outputs before final norm; chat content mask uses full token offset inclusion',special_tokens='Raw no added special tokens; chat all includes shared template/special tokens; chat content excludes template with identical forward context',sequence_rule='Reject over2048 before extraction; no truncation',exposure='FLORES dev corpus historically exposed; selected300 exclude known previous dev64/demos but cannot claim globally untouched',quality_review='24 hash-selected pairs exported for assistant spot review; not a human/native-speaker translation audit',new_training_jobs=0)
 (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['rows','eligible','exact_overlap_ids','new_training_jobs']}))
if __name__=='__main__':main()
