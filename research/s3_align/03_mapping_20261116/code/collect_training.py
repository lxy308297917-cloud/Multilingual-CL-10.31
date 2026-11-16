from common import *
import importlib.util,csv,itertools,argparse,numpy as np
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--partial',action='store_true');ap.add_argument('--knowledge',action='store_true');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 if a.knowledge:assert read(R/'analysis/formal-final/AUDIT.json')['status']=='ALL_CORE_FORMAL_COMPLETE'
 source=W/'work/bactrian-single-20261112/collect_results.py';spec=importlib.util.spec_from_file_location('s3map_upstream_audit',source);shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared);shared.LOG=LOG/'evaluation'
 tasks=[('sw','Global-MMLU',14042),('en','MMLU',14042)] if a.knowledge else [('sw','Belebele',900),('en','GSM8K',1319),('en','IFEval-prompt',541),('en','IFEval-instruction',834)];models=['Base','T0','T1','T2'];vectors={};contexts={};scores=[];provenance=[];missing=[]
 for model in models:
  if model!='Base':
   cp=CK/f'{model}-full-v1'
   if not (cp/'TRAINING_DONE.json').exists():missing.append([model,'TRAINING']);continue
   done=read(cp/'TRAINING_DONE.json');assert done['status']=='TRAINING_DONE' and done['steps']==4089 and done['counts']['samples']==65415
   digest=sha(cp/'adapter_model.safetensors');assert done['weights_sha256']==read(cp/'RELOAD_VERIFIED.json')['weights_sha256'] and digest==done['weights_sha256']['adapter_model.safetensors']
  else:digest=None
  for lang,task,n in tasks:
   stage={'Belebele':'reading','Global-MMLU':'gmmlu','MMLU':'mmlu'}.get(task,'core');folder=LOG/'evaluation'/model/stage
   if not (folder/'DONE.json').exists():missing.append([model,task]);continue
   plan,done=read(folder/'PLAN.json'),read(folder/'DONE.json');assert done['returncode']==0 and plan['adapter_sha256']==done['adapter_sha256']==digest
   assert plan['evaluation_lock']==sha(R/'EVALUATION_LOCK.json')
   vector,sources=shared.locate(model,lang,task);assert len(vector)==n
   context=shared.protocol_context(sources);args=shared.result_json(str(Path(sources[0]['path']).parent))['config']['model_args']
   assert Path(args['pretrained']).resolve()==BASE.resolve()
   if model!='Base':assert Path(args.get('peft',args.get('lora_local_path'))).resolve()==cp.resolve()
   else:assert not args.get('peft') and not args.get('lora_local_path')
   key=model,task;vectors[key]=vector;contexts[key]=context
   scores.append(dict(model=model,budget='BASE' if model=='Base' else 'FULL-65K',language=lang,task=task,n=n,score_percent=float(np.mean([r['value'] for r in vector.values()])),status='RAW_UPSTREAM_SCORE_VERIFIED'))
   provenance.append(dict(model=model,task=task,sources=sources,context=context))
 pairs=[]
 for ci,(before,after) in enumerate(itertools.combinations(models,2)):
  for ti,(lang,task,n) in enumerate(tasks):
   ka,kb=(before,task),(after,task)
   if ka not in vectors or kb not in vectors:continue
   assert contexts[ka]==contexts[kb],(before,after,task,'Protocol mismatch')
   result=shared.audit.paired(vectors[ka],vectors[kb],np.random.SeedSequence([42,ci,ti]));pairs.append(dict(before=before,after=after,language=lang,task=task,**result))
 for filename,records in [('training_results.csv',scores),('training_paired_intervals.csv',pairs)]:
  if records:
   with (a.output/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 for (model,task),v in vectors.items():
  keys=sorted(v,key=str);write(a.output/'numeric'/f'{model}_{task}.json',dict(item_ids=[str(k) for k in keys],identity=[v[k]['identity'] for k in keys],clusters=[v[k]['cluster'] for k in keys],values=[v[k]['value'] for k in keys]))
 write(a.output/'provenance.json',provenance)
 flags=[]
 for arm in ([] if a.knowledge else ['T1','T2']):
  needed=['Belebele','GSM8K','IFEval-prompt'];deltas={t:next((p['delta'] for p in pairs if p['before']=='T0' and p['after']==arm and p['task']==t),None) for t in needed}
  if all(v is not None for v in deltas.values()):flags.append(dict(model=arm,deltas=deltas,worth_followup_flag=deltas['Belebele']>=2 and deltas['GSM8K']>=-1 and deltas['IFEval-prompt']>=-1,interpretation='Predeclared engineering screen,not statistical significance or paper success'))
 write(a.output/'AUDIT.json',dict(at=now(),status='PARTIAL' if missing else 'ALL_OPTIONAL_KNOWLEDGE_COMPLETE' if a.knowledge else 'ALL_CORE_FORMAL_COMPLETE',scores=len(scores),paired_intervals=len(pairs),missing=missing,flags=flags,scoring_source=str(source),scoring_source_sha256=sha(source),audit_source_sha256=sha(shared.audit.__file__),training_lock=sha(R/'TRAINING_LOCK.json'),evaluation_lock=sha(R/'EVALUATION_LOCK.json'),limitations=['Exposed standard benchmarks,not independent confirmation','One training seed,bootstrap conditional on these checkpoints','IFEval prompt/instruction two summaries of same541prompts,not independent evidence','Unadjusted intervals; no arbitrary cross-task average','Original GSM8K extraction/numeric formatting limitations retained']))
 print(json.dumps(read(a.output/'AUDIT.json')))
 if not a.partial:assert not missing and len(scores)==(8 if a.knowledge else 16) and len(pairs)==(12 if a.knowledge else 24)
if __name__=='__main__':main()
