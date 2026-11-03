"""Freeze one source-count-matched Hausa stage; official LF tokenization only."""
import argparse,collections,hashlib,json,os,random,unicodedata,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
from datasets import Dataset,DatasetDict
from transformers import AutoTokenizer
from llamafactory.hparams import DataArguments
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--sw-manifest',type=Path,required=True);p.add_argument('--paired-dev',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);t=time.time();audit=json.loads((a.audit/'audit.json').read_text());sw=json.loads(a.sw_manifest.read_text());quota=sw['stats']['train6400']['sources'];eligible=a.audit/'eligible_candidates.jsonl';assert hashlib.sha256(eligible.read_bytes()).hexdigest()==audit['eligible_file_sha256'];rows=list(map(json.loads,eligible.open()));pools=collections.defaultdict(list);rng=random.Random(20261103)
def norm(x):return ' '.join(unicodedata.normalize('NFKC',x).casefold().split())
blocked={hashlib.sha256(norm(r['question']).encode()).hexdigest() for r in map(json.loads,a.paired_dev.open())};excluded=0
for row in rows:
 if row['prompt_sha256'] in blocked:excluded+=1;continue
 pools[row['dataset_name']].append(row)
train=[];dev=[];dev_quotas={s:23+(i<3) for i,s in enumerate(quota)}
for s,n in quota.items():
 rng.shuffle(pools[s]);nd=dev_quotas[s];assert len(pools[s])>=nd+n,(s,len(pools[s]),nd+n);dev.extend(pools[s][:nd]);train.extend(pools[s][nd:nd+n])
assert len(train)==6400 and len(dev)==256;assert collections.Counter(r['dataset_name'] for r in train)==quota
# Randomize order without breaking the exact per-family counts.
rng.shuffle(train);rng.shuffle(dev);assert not {r['prompt_sha256'] for r in train}&{r['prompt_sha256'] for r in dev}
tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True);args=DataArguments(template='qwen3',enable_thinking=False,train_on_prompt=False,cutoff_len=2048,packing=False);template=get_template_and_fix_tokenizer(tok,args);processor=SupervisedDatasetProcessor(template,tok,None,args);stats={};info={}
for name,selected in [('train6400',train),('dev_ha',dev)]:
 features=[];trunc=0
 for row in selected:
  msg=[{'role':'user','content':row['inputs']},{'role':'assistant','content':row['targets']}];ids,labels=processor._encode_data_example(msg[:1],msg[1:],None,None,[],[],[]);src,tgt=template.encode_multiturn(tok,msg,None,None)[0];trunc+=len(src)+len(tgt)>2048;assert len(ids)==len(labels)<=2048 and any(x==-100 for x in labels) and sum(x!=-100 for x in labels[1:])>0;features.append({'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels})
  if len(features)%1000==0:print(json.dumps({'split':name,'encoded':len(features)}),flush=True)
 (a.output/(name+'.jsonl')).write_text(''.join(json.dumps({'instruction':r['inputs'],'output':r['targets']},ensure_ascii=False)+'\n' for r in selected));(a.output/(name+'.provenance.jsonl')).write_text(''.join(json.dumps({k:v for k,v in r.items() if k not in ['inputs','targets']},ensure_ascii=False)+'\n' for r in selected));DatasetDict({'train':Dataset.from_list(features)}).save_to_disk(str(a.output/(name+'_tokenized')));stats[name]={'rows':len(selected),'input_tokens':sum(len(f['input_ids']) for f in features),'shifted_assistant_tokens':sum(sum(v!=-100 for v in f['labels'][1:]) for f in features),'truncated':trunc,'sources':dict(collections.Counter(r['dataset_name'] for r in selected))};info[name]={'file_name':name+'.jsonl','columns':{'prompt':'instruction','response':'output'}}
(a.output/'dataset_info.json').write_text(json.dumps(info,indent=2)+'\n');m={'purpose':'Stage2 Hausa6400, fixed same source-family counts as original Stage1 SW6400, not claimed token/content matched','source_audit_sha256':hashlib.sha256((a.audit/'audit.json').read_bytes()).hexdigest(),'source_revision':audit['source_revision'],'selection_seed':20261103,'dev_rows':256,'known_dev_exact_prompt_exclusions':excluded,'source_counts_match_SW6400':True,'same_semantic_items_as_SW':False,'pairing_warning':'Collection numeric IDs coincide with per-file row positions; do not use as certified cross-language semantic keys','stats':stats,'stage1_SW_token_counts':{k:sw['stats']['train6400'][k] for k in ['input_tokens','shifted_assistant_tokens']},'template':'official LF qwen3, nonthinking, assistant-only, cutoff2048,no packing','quality_limitations':audit['quality_limits'],'formal_test_used':False,'model_outputs_used':False,'training_started':False,'seconds':time.time()-t,'files':{str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()}};(a.output/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');print(json.dumps(m,ensure_ascii=False),flush=True)
