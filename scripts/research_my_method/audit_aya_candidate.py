"""One bounded language-source audit, using frozen SW source families as quotas."""
import argparse,collections,hashlib,json,os,random,subprocess,unicodedata,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
import pyarrow.parquet as pq
p=argparse.ArgumentParser();p.add_argument('--source-lock',type=Path,required=True);p.add_argument('--sw-manifest',type=Path,required=True);p.add_argument('--language',required=True);p.add_argument('--lid-code',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);lock=json.loads(a.source_lock.read_text());source=Path(lock['path']);assert source.stat().st_size==lock['bytes'];sw=json.loads(a.sw_manifest.read_text());quota=sw['stats']['train6400']['sources'];families=list(quota);rng=random.Random(20261103);pools={s:[] for s in families};counts=collections.Counter();total=0;t=time.time()
def norm(x):return ' '.join(unicodedata.normalize('NFKC',x).casefold().split())
for batch in pq.ParquetFile(source).iter_batches(batch_size=8192):
 for row in batch.to_pylist():
  idx=total;total+=1;s=row['dataset_name'];counts[s]+=1
  if s not in pools:continue
  row['source_index']=idx;pool=pools[s]
  if len(pool)<1400:pool.append(row)
  else:
   k=rng.randrange(counts[s])
   if k<len(pool):pool[k]=row
candidates=[row for s in families for row in pools[s]];raw=a.output/'candidate_inputs.jsonl';raw.write_text(''.join(json.dumps({'inputs':r['inputs'],'targets':r['targets']},ensure_ascii=False)+'\n' for r in candidates));program='''import fasttext,json,sys
m=fasttext.load_model('/local/models/glotlid/model_v3.bin')
for line in open(sys.argv[1]):
 r=json.loads(line);out={}
 for key in ['inputs','targets']:
  labs,ps=m.predict(r[key].replace('\\n',' ')[:6000],k=3);out[key]=list(zip(labs,map(float,ps)))
 print(json.dumps(out))
''';proc=subprocess.run(['/local/lid-env/bin/python','-c',program,str(raw)],capture_output=True,text=True,check=True);(a.output/'candidate_lid.jsonl').write_text(proc.stdout);lids=[json.loads(l) for l in proc.stdout.splitlines()];assert len(lids)==len(candidates);rejected=collections.Counter();eligible=collections.defaultdict(list);seen=set();lengths=[]
for row,lid in zip(candidates,lids):
 x,y=row['inputs'],row['targets'];nx,ny=norm(x),norm(y);key=hashlib.sha256(nx.encode()).hexdigest()
 if not nx or not ny or '\ufffd' in x+y or '<unk>' in x+y or len(x)>24000 or len(y)>14000:rejected['empty_corrupt_or_extreme']+=1;continue
 if key in seen:rejected['duplicate_instruction']+=1;continue
 if len(x.split())>=15 and a.lid_code not in lid['inputs'][0][0]:rejected['instruction_lid']+=1;continue
 if len(y.split())>=5 and a.lid_code not in lid['targets'][0][0]:rejected['response_lid']+=1;continue
 seen.add(key);row['prompt_sha256']=key;row['pair_sha256']=hashlib.sha256((nx+'\x1f'+ny).encode()).hexdigest();row['lid']=lid;eligible[row['dataset_name']].append(row);lengths.append(len(x)+len(y))
counts_eligible={s:len(eligible[s]) for s in families};deficits={s:quota[s]-len(eligible[s]) for s in families if len(eligible[s])<quota[s]};eligible_file=a.output/'eligible_candidates.jsonl';eligible_file.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for s in families for r in eligible[s]));review=[]
for s in families:
 rng.shuffle(eligible[s]);review.extend(eligible[s][:3])
(a.output/'review_samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review));report={'source_lock_sha256':hashlib.sha256(a.source_lock.read_bytes()).hexdigest(),'source_revision':lock['revision'],'language':a.language,'total_source_rows':total,'all_source_counts':dict(counts),'frozen_SW_train6400_source_quotas':quota,'reservoir_per_family':1400,'bounded_candidates':len(candidates),'eligible_counts':counts_eligible,'eligible_total':sum(counts_eligible.values()),'rejected':dict(rejected),'exact_source_quota_deficits':deficits,'can_match_SW6400_source_counts_before_dev_reservation':not deficits,'eligible_file_sha256':hashlib.sha256(eligible_file.read_bytes()).hexdigest(),'seconds':time.time()-t,'status':'DATA_FEASIBILITY_ONLY','quality_limits':['Same bounded GlotLID/length/dedup policy as original SW; LID is fallible and short answers are exempt.','Source family matching is not semantic task matching.','No full-corpus contamination audit; known benchmark source families omitted by frozen SW whitelist.','Independent bilingual human quality review not performed.','No test scores or model outputs used for language choice.']};(a.output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False),flush=True)
