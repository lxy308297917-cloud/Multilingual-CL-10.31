"""Independent audit of the five preregistered tied-embedding interventions."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for k in ['project','run','cohort','registration','baseline-run','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
sys.path[:0]=[str(a.project/'scripts/research_my_method'),str(a.project/'scripts/bench_v2')]
from analyze_coefficient_pilot import audit_rows
from paired_statistics import paired_macro
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
done=json.loads((a.run/'DONE.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text());reg=json.loads(a.registration.read_text());cohort=json.loads(a.cohort.read_text())
assert done['status']=='EXPLORATORY_EMBEDDING_CONTROL_COMPLETE' and done['predictions_sha256']==sha(a.run/'predictions.jsonl') and done['protocol_sha256']==sha(a.run/'protocol.json')
assert protocol['registration_sha256']==sha(a.registration) and protocol['cohort_protocol_sha256']==sha(a.cohort)==reg['cohort_protocol_sha256']
assert protocol['conditions']==reg['interventions'] and protocol['no_optimizer_or_model_write']
groups=audit_rows(a.run/'predictions.jsonl',cohort,'condition');assert set(groups)==set(reg['interventions']) and sum(map(len,groups.values()))==1710
base={(x['semantic_id'],x['language']):x for x in map(json.loads,(a.baseline_run/'predictions.jsonl').open())}
for key,row in groups['unchanged'].items():
 assert row['gold']==base[key]['gold'] and row['predicted']==base[key]['predicted'] and row['option_scores']==base[key]['option_scores']
assert done['baseline_max_absolute_logit_difference']==0
interventions=[json.loads(x) for x in (a.run/'interventions.jsonl').open()];assert len(interventions)==4
for x in interventions:
 assert x['tied_roles_preserved'] and x['other_parameters_frozen'] and x['max_row_relative_norm_error']<1e-4
 assert abs(x['global_effective_delta_norm']-x['alpha']*protocol['delta_squared_norm']**.5)<1e-6
scores={};vectors={}
for name,rows in groups.items():
 scores[name]={};vectors[name]={}
 for lang in cohort['languages']:
  v=np.array([rows[sid,lang]['correct'] for sid in cohort['confirmation_ids']],dtype=float);correct=int(v.sum());vectors[name][lang]=100*v
  assert done['results'][name][lang]['n']==len(v) and done['results'][name][lang]['correct']==correct
  scores[name][lang]={'n':len(v),'correct':correct,'acc':correct/len(v)}
pairs=[('restore_tied_embedding_alpha1.0','random_sign_matched_alpha1.0',True),('restore_tied_embedding_alpha1.0','unchanged',False),('random_sign_matched_alpha1.0','unchanged',False),('restore_tied_embedding_alpha0.5','random_sign_matched_alpha0.5',False),('restore_tied_embedding_alpha0.5','unchanged',False),('random_sign_matched_alpha0.5','unchanged',False)]
contrasts=[]
for t,b,primary in pairs:
 for lang in cohort['languages']:
  result=paired_macro(vectors[t][lang],vectors[b][lang],10000,20261103);result['unit']='paired original semantic IDs'
  contrasts.append({'treatment':t,'control':b,'language':lang,'primary_exploratory':primary and lang in ['en','sw'],**result})
report={'status':'VERIFIED_EXPLORATORY_CONTROL_COMPLETE','predictions_verified':1710,'scores':scores,'contrasts':contrasts,'baseline_logits_exact':True,'all_four_effective_norm_controls_verified':True,'registration_sha256':sha(a.registration),'cohort_sha256':sha(a.cohort),'source_predictions_sha256':sha(a.run/'predictions.jsonl'),'source_protocol_sha256':sha(a.run/'protocol.json'),'analysis_code_sha256':sha(Path(__file__)),'formal_benchmark':False,'adoption_claim_allowed':False,
 'limitations':['Historically exposed knowledge dev,114 semantic IDs, unadjusted intervals.','Jointly tied input/output intervention cannot identify independent roles.','Conditional Final perturbation does not reconstruct training cause.','Energy is not component importance; no cross-component matched test.','No IFEval/GSM/code protection demonstrated.']}
(a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# tied embedding有限恢复：探索性结果','','1710条预测的gold、argmax、同模型baseline精确logits及四个实际范数控制独立复核通过。','', '| 条件 | EN知识dev ACC | SW知识dev ACC | HA知识dev ACC |','|---|---:|---:|---:|']
for name in reg['interventions']:lines.append('| '+name+' | '+' | '.join('%.2f%%'%(100*scores[name][x]['acc']) for x in cohort['languages'])+' |')
lines+=['','## 预登记主要对比：完整恢复减同范数随机符号','', '| 语言 | ACC差pp | 95%配对区间 |','|---|---:|---:|']
for x in contrasts:
 if x['primary_exploratory']:lines.append('| %s | %+.2f | [%+.2f, %+.2f] |'%(x['language'],x['delta'],x['ci95_low'],x['ci95_high']))
lines+=['','不能据96.21%能量直接称embedding造成全部遗忘。这里的EN知识在适配后本就较Base提高，不是发生IFEval/GSM遗忘的同一任务。恢复未显示明确EN/SW改善，随机符号扰动也不代表完整语义空对照。输入/输出角色仍绑在一起；不是新方法或正式benchmark。']
(a.output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'predictions_verified':1710,'primary':[x for x in contrasts if x['primary_exploratory']]}))
