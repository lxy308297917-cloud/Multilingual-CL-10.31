"""Freeze semantic-ID splits from official dev only, before new model outputs."""
import argparse,collections,hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);langs=['en','sw','ha','yo'];data={l:pq.read_table(a.source/(l+'-dev.parquet')).to_pylist() for l in langs};maps={l:{r['sample_id']:r for r in rows} for l,rows in data.items()};ids=set(maps['en']);assert len(ids)==285
for l in langs:assert set(maps[l])==ids and len(maps[l])==len(data[l])==285
by=collections.defaultdict(list);unchanged_questions=collections.Counter();cultural=collections.Counter()
for sid in sorted(ids):
 gold=maps['en'][sid]['answer'];subject=maps['en'][sid]['subject'];assert gold in 'ABCD';by[subject].append(sid)
 for l in langs:
  r=maps[l][sid];assert r['answer']==gold and r['subject']==subject
  assert all(isinstance(r[k],str) and r[k].strip() for k in ['question','option_a','option_b','option_c','option_d'])
  unchanged_questions[l]+=r['question']==maps['en'][sid]['question'];cultural[l]+=bool(r['is_annotated'])
splits={};seed='multilingual-cl-stage2-dev-20261103-v1'
for subject,group in sorted(by.items()):
 assert len(group)==5
 ordered=sorted(group,key=lambda sid:hashlib.sha256((seed+'|'+sid).encode()).hexdigest())
 for i,sid in enumerate(ordered):splits[sid]='fit' if i<3 else 'confirmation'
manifest={'dataset':'CohereLabs/Global-MMLU','revision':'0e619dbeb34206cd48705a1a0ea7fb21cae09993','source_split':'dev','languages':langs,'semantic_ids':285,'subjects':len(by),'split_counts':dict(collections.Counter(splits.values())),'split_rule':'Within each subject, SHA256(seed|sample_id) order, first3 fit and last2 confirmation; all translations of an ID stay together','seed':seed,'source_hashes':{l:hashlib.sha256((a.source/(l+'-dev.parquet')).read_bytes()).hexdigest() for l in langs},'known_prior_exposure':'EN/SW dev285 were used in earlier checkpoint selection/few-shot contexts. These are not historically pristine holdouts. New candidate fit/confirmation split is frozen before new method outputs. Hausa/Yoruba dev not previously evaluated in this project to current audit knowledge.','quality':{'sample_id_subject_gold_labels_consistent':True,'human_semantic_translation_review':False,'semantic_pairing_status':'official shared semantic IDs and option labels verified; translation semantics not independently certified','cultural_annotation_true_counts':dict(cultural),'cultural_annotation_is_not_translation_quality':True,'question_identical_to_english_counts':dict(unchanged_questions)},'formal_test_used':False,'may_select_new_method_as_confirmed':False,'allowed_use':'exploratory task-format-aligned mechanism pilot; independent fresh confirmation required for an adoption claim'}
rows=[]
for sid in sorted(ids):
 for lang in langs:
  r=maps[lang][sid];rows.append({'id':lang+'|'+sid,'semantic_id':sid,'split':splits[sid],'language':lang,'subject':r['subject'],'question':r['question'],'choices':[r['option_'+x] for x in 'abcd'],'answer':r['answer'],'source_split':'dev','source_revision':manifest['revision'],'translation_semantics_independently_verified':False})
f=a.output/'paired_dev.jsonl';f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));manifest['paired_dev_sha256']=hashlib.sha256(f.read_bytes()).hexdigest();(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(a.output/'split_ids.json').write_text(json.dumps(splits,indent=2)+'\n');print(json.dumps(manifest),flush=True)
