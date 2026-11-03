"""Bounded read-only source/label profile. No translation-quality certification."""
from pathlib import Path
import argparse,json,hashlib,re,collections
from datasets import load_from_disk
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);d=a.dataset;sources={}
def rows(path):sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest();return list(map(json.loads,path.open()))
summary={};allpairs={}
for split in ['train6400','dev_ha']:
 raw=rows(d/(split+'.jsonl'));prov=rows(d/(split+'.provenance.jsonl'));tok=load_from_disk(str(d/(split+'_tokenized')))['train'];assert len(raw)==len(prov)==len(tok);counts=collections.Counter();nll_counts=collections.Counter();labels=collections.Counter();tasks=collections.Counter();lid=collections.Counter();lid_by_length=collections.Counter();rawpairs=set();truncated=0;inputtokens=0;supervised=0;by={}
 for x,y,z in zip(raw,prov,tok):
  assert x['instruction'].strip() and x['output'].strip();pair=(x['instruction'],x['output']);rawpairs.add(pair);src=y['dataset_name'];counts[src]+=1;tasks[y['task_type']]+=1;n=sum(v!=-100 for v in z['labels'][1:]);assert n>0;assert len(z['input_ids'])==len(z['labels']);supervised+=n;inputtokens+=len(z['input_ids']);nll_counts[src]+=n;truncated+=len(z['input_ids'])==2048
  label=x['output'].strip();shape='single_ABCD' if re.fullmatch(r'[ABCDabcd][.)]?',label) else 'short_non_letter_1_to_4_words' if len(label.split())<=4 else 'longer_text';labels[shape]+=1
  q=by.setdefault(src,{'rows':0,'assistant_tokens':0,'single_ABCD':0,'truncated_length2048':0});q['rows']+=1;q['assistant_tokens']+=n;q['single_ABCD']+=shape=='single_ABCD';q['truncated_length2048']+=len(z['input_ids'])==2048
  for field in ['inputs','targets']:
   preds=y.get('lid',{}).get(field)
   if preds:
    lid[field+':'+preds[0][0]]+=1
    text=x['instruction'] if field=='inputs' else x['output'];threshold=15 if field=='inputs' else 5;length='long_checked' if len(text.split())>=threshold else 'short_exempt';language='HA_top' if preds[0][0]=='__label__hau_Latn' else 'other_top';lid_by_length[field+':'+length+':'+language]+=1
 allpairs[split]=rawpairs;summary[split]={'rows':len(raw),'unique_exact_instruction_output_pairs':len(rawpairs),'input_tokens':inputtokens,'shifted_assistant_tokens':supervised,'source_families':dict(counts),'task_types':dict(tasks),'reference_shapes':dict(labels),'rows_at_cutoff2048':truncated,'translated_T_source_rows':sum(v for k,v in counts.items() if k.endswith('(T)')),'saved_LID_top_labels':dict(lid),'saved_LID_by_original_length_policy':dict(lid_by_length),'by_source':by}
assert not allpairs['train6400']&allpairs['dev_ha'];curves={}
root=a.project/'results/next_stage/dev_curves_v1';devprov=rows(d/'dev_ha.provenance.jsonl')
for group in ['fft','sst','ha_only']:
 for step in [0,100,200,300,400]:
  node=root/group/f'step{step:03d}'
  if step==0 and group=='ha_only':node=root/'base/step000'
  f=node/'nll_predictions.jsonl'
  if not f.exists():continue
  totals={}
  for v in map(json.loads,f.open()):
   i=int(v['id'].rsplit('-',1)[1]);src=devprov[i]['dataset_name'];q=totals.setdefault(src,{'nll_sum':0.,'assistant_tokens':0,'rows':0});q['nll_sum']+=v['nll_sum'];q['assistant_tokens']+=v['assistant_tokens'];q['rows']+=1
  for q in totals.values():q['token_weighted_nll']=q['nll_sum']/q['assistant_tokens']
  curves[group+'/'+str(step)]=totals
res={'status':'READ_ONLY_PROFILE_COMPLETE','split_summary':summary,'exact_train_heldout_pair_overlap':0,'heldout_nll_by_source':curves,'source_file_sha256':sources,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limits':['Saved GlotLID scores are classifier outputs, not certified probabilities or bilingual semantic accuracy.','A (T) source tag identifies translated origin, not a demonstrated mistranslation.','Reference-shape mismatch is descriptive, not proof of the cause of benchmark changes.','Exact-pair disjointness does not certify absence of semantic overlap.','No examples or per-item records are published.']};(a.output/'analysis.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:{z:v[z] for z in ['rows','shifted_assistant_tokens','task_types','reference_shapes','translated_T_source_rows']} for k,v in summary.items()}))
