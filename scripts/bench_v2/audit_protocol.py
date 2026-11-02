"""Read-only dataset identity, split, URL overlap, and exact contamination audit."""
import argparse,json,hashlib,unicodedata,collections
from pathlib import Path
from common import jsonl,ids_hash,sha256,dump


def verify_rows(rows,manifest):
 ids=[r['id'] for r in rows]
 if not rows or len(set(ids))!=len(ids) or len(rows)!=manifest['samples']:raise ValueError('Unexpected samples or duplicate IDs')
 if ids_hash(rows)!=manifest['sample_ids_sha256']:raise ValueError('ID order/hash differs')
 if any(r['split']!=manifest['dataset_split'] for r in rows):raise ValueError('Wrong dataset split')
 if any(r['direction']!=manifest['language'] for r in rows):raise ValueError('Wrong task direction')
 return True


def norm(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())

def exact_overlap(train_texts,benchmark_rows):
 # Exact normalized full fields only; does not prove absence of partial/paraphrased overlap.
 lookup=collections.defaultdict(list)
 for id,text in train_texts:
  normalized=norm(text)
  if normalized:lookup[hashlib.sha256(normalized.encode()).hexdigest()].append(str(id))
 hits=[]
 for row in benchmark_rows:
  for field in ['source','reference']:
   if not isinstance(row.get(field),str) or not norm(row[field]):continue
   key=hashlib.sha256(norm(row[field]).encode()).hexdigest()
   if key in lookup:hits.append(dict(id=row['id'],field=field,training_ids=lookup[key]))
 return hits


def crosssum_overlap(cross,source_xl,target_xl):
 source_urls={r['url'] for r in source_xl};target_urls={r['url'] for r in target_xl};source_texts={hashlib.sha256(norm(r['source']).encode()).hexdigest() for r in source_xl};target_refs={hashlib.sha256(norm(r['reference']).encode()).hexdigest() for r in target_xl}
 return dict(samples=len(cross),source_url_shared_with_xlsum_test=sum(r['source_url'] in source_urls for r in cross),target_url_shared_with_xlsum_test=sum(r['target_url'] in target_urls for r in cross),exact_source_text_shared_with_xlsum_test=sum(hashlib.sha256(norm(r['source']).encode()).hexdigest() in source_texts for r in cross),exact_target_summary_shared_with_xlsum_test=sum(hashlib.sha256(norm(r['reference']).encode()).hexdigest() in target_refs for r in cross),scope='Against official XL-Sum test splits only; source or target may also share XL-Sum train/val, not counted here',semantic_alignment_verified=False)


def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--training-json',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);bundle=json.loads((a.data/'manifest.json').read_text());datasets={};info={}
 for name,cfg in bundle.items():
  rows=jsonl(a.data/name/'test.jsonl');verify_rows(rows,cfg)
  if sha256(a.data/name/'test.jsonl')!=cfg['data_sha256']:raise ValueError('Data content hash differs')
  datasets[name]=rows;info[name]=dict(samples=len(rows),ids_unique=True,split=cfg['dataset_split'],data_sha256=cfg['data_sha256'],empty_sources=sum(not r['source'].strip() for r in rows))
 overlap={}
 for src,tgt in [('en','sw'),('sw','en')]:overlap[src+'->'+tgt]=crosssum_overlap(datasets[f'crosssum_{src}_{tgt}_v2'],datasets[f'xlsum_{src}_v2'],datasets[f'xlsum_{tgt}_v2'])
 contamination=None
 if a.training_json:
  raw=json.loads(a.training_json.read_text()) if a.training_json.suffix=='.json' else jsonl(a.training_json);texts=[]
  for index,row in enumerate(raw):
   if 'conversations' in row:values=[m.get('value','') for m in row['conversations']]
   elif 'messages' in row:values=[m.get('content','') for m in row['messages']]
   else:values=[v for k,v in row.items() if k in ['instruction','input','output','prompt','response'] and isinstance(v,str)]
   texts.extend((str(index),v) for v in values if v)
  if not texts:raise ValueError('Training schema unrecognized; cannot claim contamination audit')
  contamination=dict(training_samples=len(raw),training_text_fields=len(texts),training_sha256=sha256(a.training_json),hits={name:exact_overlap(texts,rows) for name,rows in datasets.items()},scope='Normalized exact complete user/assistant field matches only; not a broad decontamination or proof of unseen text. Official test IDs are retained unchanged.')
 dump(a.output/'audit.json',dict(datasets=info,crosssum_xlsum_overlap=overlap,exact_training_overlap=contamination));print(json.dumps(dict(dataset_counts={k:v['samples'] for k,v in info.items()},crosssum_xlsum_overlap=overlap)))
if __name__=='__main__':main()
