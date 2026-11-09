"""Independently audit likelihood and real execution outputs, then pair frozen rows."""
from pathlib import Path
import collections,csv,hashlib,itertools,json
import numpy as np
R=Path('/data0/users/lixiaoyang/10.31/multilingual-research')
L=Path('/data0/users/lixiaoyang/logs/qwen3-multilingual/representation-benchmarks-20261109')
D=Path('/data1/lixiaoyang/datasets/representation-benchmarks-20261109/v1')
N=R/'research/representation_benchmarks_20261109'
MODELS=['Base','all_layer','full_feature','ce_replay']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readrows(p):return [json.loads(x) for x in p.open()]
def csvwrite(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def only(p,pattern):
 xs=list(p.rglob(pattern));assert len(xs)==1,(p,pattern,len(xs));return xs[0]
def paired(a,b,groups,seed=42):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);groups=np.asarray(groups);assert a.shape==b.shape==groups.shape
 unique,inverse=np.unique(groups,return_inverse=True);n=len(unique);sums=np.bincount(inverse,weights=b-a,minlength=n);sizes=np.bincount(inverse,minlength=n)
 rng=np.random.default_rng(seed);boot=[]
 for _ in range(40):
  idx=rng.integers(n,size=(50,n));boot.extend(sums[idx].sum(1)/sizes[idx].sum(1)*100)
 lo,hi=np.quantile(boot,[.025,.975])
 return dict(delta_percentage_points=float((b-a).mean()*100),paired95_low=float(lo),paired95_high=float(hi),base_correct_adapted_wrong=int(((a==1)&(b==0)).sum()),base_wrong_adapted_correct=int(((a==0)&(b==1)).sum()),samples=len(a),bootstrap_clusters=n,bootstrap_resamples=2000)
def main():
 out=N/'benchmarks';out.mkdir(parents=True,exist_ok=False)
 assets=json.loads((R/'configs/representation_benchmarks_20261109/assets.json').read_text());scores=[];items=[];prefs=[];vectors={};clusters={};source=[]
 for m in MODELS:
  root=L/('formal-'+m.lower()+'-xnli');assert (root/'DONE.json').exists()
  resultfile=only(root/'results','results_*.json');result=json.loads(resultfile.read_text());assert int(result['config']['batch_size'])==4 and result['config']['model']=='hf'
  ma=result['config']['model_args'];ma=ma if isinstance(ma,dict) else dict(x.split('=',1) for x in ma.split(','))
  assert ma['pretrained']==assets['models'][m]['base_path'] and ma.get('peft')==assets['models'][m]['adapter']
  assert ma['dtype']=='bfloat16' and str(ma['enable_thinking']).lower()=='false'
  for lang,kind in itertools.product(['sw','en'],['primary','rotated']):
   name='rb_xnli_'+lang+'_formal_'+kind;samplefile=only(root/'results','samples_'+name+'_*.jsonl');rows=readrows(samplefile);frozen=readrows(D/'xnli'/(lang+'_formal_'+kind+'.jsonl'));assert len(rows)==len(frozen)==5010
   rows=sorted(rows,key=lambda x:int(x['doc']['id']));assert [x['doc'] for x in rows]==frozen
   preds=[];correct=[];semantic=[];gold=[]
   for x in rows:
    ll=[float(v[0]) for v in x['filtered_resps']];assert len(ll)==3 and all(np.isfinite(ll));p=int(np.argmax(ll));g=x['doc']['label'];c=int(p==g);assert x['acc']==c
    sem=x['doc']['semantic_at_letter'][p];gsem=x['doc']['original_label'];assert c==int(sem==gsem)
    items.append(dict(model=m,language=lang,protocol=kind,id=x['doc']['id'],premise_cluster=x['doc']['premise_cluster_sha256'],gold_letter=g,predicted_letter=p,gold_semantic=gsem,predicted_semantic=sem,correct=c,loglikelihood_A=ll[0],loglikelihood_B=ll[1],loglikelihood_C=ll[2],correct_answer_margin=ll[g]-max(v for i,v in enumerate(ll) if i!=g)))
    preds.append(p);correct.append(c);semantic.append(sem);gold.append(gsem)
   assert abs(np.mean(correct)-result['results'][name]['acc,none'])<1e-12
   vectors[m,lang,kind]=correct;clusters[lang,kind]=[x['doc']['premise_cluster_sha256'] for x in rows]
   scores.append(dict(model=m,task='XNLI '+lang.upper(),metric='ACC',protocol=kind,score=float(np.mean(correct)*100),samples=5010,unit='percent'))
   matrix=[[sum(g==i and p==j for g,p in zip(gold,semantic)) for j in range(3)] for i in range(3)]
   prefs.append(dict(model=m,language=lang,protocol=kind,semantic_labels=['entailment','neutral','contradiction'],confusion_true_rows_predicted_columns=matrix,per_semantic_class_acc=[matrix[i][i]/sum(matrix[i])*100 for i in range(3)],predicted_letter_counts=[preds.count(i) for i in range(3)],predicted_semantic_counts=[semantic.count(i) for i in range(3)],max_letter_share=max(preds.count(i) for i in range(3))/5010,max_semantic_share=max(semantic.count(i) for i in range(3))/5010))
   source.append(dict(model=m,task=name,raw_samples_sha256=sha(samplefile),aggregate_sha256=sha(resultfile)))
 contrasts=[]
 for lang,kind in itertools.product(['sw','en'],['primary','rotated']):
  for a,b in [('Base','all_layer'),('Base','full_feature'),('Base','ce_replay'),('all_layer','full_feature'),('ce_replay','full_feature')]:
   contrasts.append(dict(task='XNLI '+lang.upper(),protocol=kind,reference=a,comparison=b,bootstrap_unit='shared premise cluster',**paired(vectors[a,lang,kind],vectors[b,lang,kind],clusters[lang,kind])))
  for m in MODELS:contrasts.append(dict(task='XNLI '+lang.upper(),protocol='primary_to_fixed_rotation',reference=m+':primary',comparison=m+':rotated',bootstrap_unit='shared premise cluster',**paired(vectors[m,lang,'primary'],vectors[m,lang,'rotated'],clusters[lang,'primary'])))
 codeitems=[];codevec={};codeaudit=[]
 for m in MODELS:
  root=L/('formal-'+m.lower()+'-code');execution=root/'execution';assert (execution/'DONE.json').exists()
  s=json.loads((execution/'scores.json').read_text());rows=json.loads((execution/'model_evaluation-results.json').read_text());rows=sorted(rows,key=lambda x:int(x['task_id'].split('/')[-1]));assert len(rows)==164 and all(x['infrastructure_status']=='COMPLETE' for x in rows)
  generation=readrows(root/'generation/predictions.jsonl');byid={x['id']:x for x in generation};assert len(byid)==164
  for key,metric in [('base','HumanEval'),('plus','HumanEval+')]:
   vec=[int(x[key+'_pass']) for x in rows];assert sum(vec)==s[key+'_correct'];codevec[m,key]=vec
   scores.append(dict(model=m,task=metric,metric='pass@1',protocol='greedy_n1_isolated_evalplus',score=float(np.mean(vec)*100),samples=164,unit='percent'))
  static=json.loads((execution/'static/static_per_sample.json').read_text());summary=json.loads((execution/'static/summary.json').read_text())
  codeaudit.append(dict(model=m,official_base_status_counts=dict(collections.Counter(x['base'] for x in rows)),official_plus_status_counts=dict(collections.Counter(x['plus'] for x in rows)),static_diagnostics=summary['static_diagnostics'],static_helper_caveat='Legacy static-only helper emits an obsolete NOT_RUN execution explanation. It does not execute candidates; actual scores and COMPLETE statuses here come from the subsequent verified isolated executor, not that static summary.',generation_hit_cap=sum(x['generation_hit_cap'] for x in generation),generated_tokens=sum(x['generated_tokens'] for x in generation),canonical_calibration='164/164 base and plus pass; exact executor identity reused after first model',static_details_sha256=sha(execution/'static/static_per_sample.json'),runtime_error_subtype='Official fast_check fail does not distinguish assertion, runtime exception or syntax; static syntax is separate, no fabricated subtype counts'))
  for x in rows:codeitems.append(dict(model=m,id=x['task_id'],base_pass=int(x['base_pass']),plus_pass=int(x['plus_pass']),base_status=x['base'],plus_status=x['plus'],solution_sha256=x['solution_sha256'],wall_seconds=x['execution']['wall_seconds']))
  source.append(dict(model=m,task='HumanEval+',raw_samples_sha256=sha(root/'generation/predictions.jsonl'),aggregate_sha256=sha(execution/'scores.json')))
 for key,metric in [('base','HumanEval'),('plus','HumanEval+')]:
  for a,b in [('Base','all_layer'),('Base','full_feature'),('Base','ce_replay'),('all_layer','full_feature'),('ce_replay','full_feature')]:contrasts.append(dict(task=metric,protocol='greedy_n1_isolated_evalplus',reference=a,comparison=b,bootstrap_unit='code task',**paired(codevec[a,key],codevec[b,key],[str(i) for i in range(164)])))
 csvwrite(out/'new_scores.csv',scores);csvwrite(out/'xnli_items.csv',items);csvwrite(out/'code_items.csv',codeitems);csvwrite(out/'paired_changes.csv',contrasts)
 for name,obj in [('xnli_label_bias.json',prefs),('code_execution_audit.json',codeaudit),('source_hashes.json',source)]: (out/name).write_text(json.dumps(obj,indent=2)+'\n')
 old=list(csv.DictReader((R/'research/functional_representation_20261109/p2/scores.csv').open()));merged=[x for x in old if x['model'] in MODELS]
 main=[x for x in scores if x['protocol']=='primary' or x['task']=='HumanEval+'];mapping={}
 for x in merged: mapping.setdefault((x['task'],x['metric'],x['unit']),{})[x['model']]=float(x['score'])
 for x in main:mapping.setdefault((x['task'],x['metric'],x['unit']),{})[x['model']]=x['score']
 table=[dict(task=task,metric=metric,unit=unit,**{m:values[m] for m in MODELS}) for (task,metric,unit),values in mapping.items()]
 csvwrite(N/'complete_scores.csv',table)
 (out/'DONE.json').write_text(json.dumps(dict(status='COMPLETE',xnli_primary_rows=40080,xnli_auxiliary_rotation_rows=40080,real_generated_executed_code=656,models=MODELS,new_training_jobs=0,bootstrap_note='2000 paired resamples; XNLI shared premise clusters, code tasks; intervals describe this fixed seed42 model, no cross-training-seed claim'),indent=2)+'\n');print(json.dumps(table,indent=2))
if __name__=='__main__':main()
