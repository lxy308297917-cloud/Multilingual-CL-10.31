"""Post hoc CPU fit-only stability; no new predictions, tuning or adoption."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser()
for k in ['project','run','bank','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);t=time.time();torch.set_num_threads(4)
sys.path.insert(0,str(a.project/'scripts/research_my_method'));from coefficient_geometry import centered_cross_space
protocol=json.loads((a.run/'protocol.json').read_text());fit=protocol['actual_fit_ids'];categories=[protocol['categories'][sid] for sid in fit];assert len(fit)==18
reg={'registered_at':__import__('datetime').datetime.now().astimezone().isoformat(),'scope':'Post hoc descriptive fit-only bootstrap, after full pilot scores were read; not an ex ante confirmation','bootstrap_repeats':100,'seed':20261103,'selection_or_predictions_changed':False,'resampling':'Three paired semantic rows with replacement independently within each of six fixed categories, same draw for every matrix','evidence_level':'Statistical/numerical diagnostic only'}
(a.output/'plan.json').write_text(json.dumps(reg,indent=2)+'\n')
g=torch.load(a.run/'fit-coefficient-gradients.pt',map_location='cpu',weights_only=True)
b=torch.load(a.bank,map_location='cpu',weights_only=True);modes=b['modes'];rng=np.random.default_rng(reg['seed']);groups=[[i for i,c in enumerate(categories) if c==cat] for cat in sorted(set(categories))];assert all(len(x)==3 for x in groups)
draws=[]
for _ in range(reg['bootstrap_repeats']):
 idx=list(range(len(fit)))
 for ids in groups:
  for j,k in zip(ids,rng.choice(ids,len(ids),replace=True).tolist()):idx[j]=k
 draws.append(idx)
rows=[]
for name in sorted(g['en']):
 en,sw=g['en'][name],g['sw'][name];q,vals=centered_cross_space(en,sw,categories);rank=q.shape[1];assert rank==2
 overlaps=[];ranks=[]
 for idx in draws:
  qb,_=centered_cross_space(en[idx],sw[idx],categories);ranks.append(qb.shape[1]);overlaps.append(float((q.double().T@qb.double()).square().sum()/rank))
 s=modes[name]['s'].double();gaps=(s[:-1]-s[1:])/s[:-1]
 assert min(overlaps)>=-1e-5 and max(overlaps)<=1+1e-4
 rows.append({'name':name,'coefficient_dimension':en.shape[1],'reference_rank':rank,'bootstrap_overlap_mean':float(np.mean(overlaps)),'bootstrap_overlap_median':float(np.median(overlaps)),'bootstrap_overlap_q025_q975':np.quantile(overlaps,[.025,.975]).tolist(),'rank_below2_count':sum(k<2 for k in ranks),'random_orientation_expected_overlap':float(np.mean(ranks)/en.shape[1]),'minimum_adjacent_relative_singular_gap':float(gaps.min()),'adjacent_relative_singular_gap_below1e-3_count':int((gaps<1e-3).sum())})
summary={'matrices':len(rows),'fit_semantic_ids':18,'independent_semantic_samples_not_rotations':18,'diagonal_coefficients':sum(x['coefficient_dimension'] for x in rows),'full_old_row_column_space_coefficients':sum(x['coefficient_dimension']**2 for x in rows),'median_matrix_mean_bootstrap_overlap':float(np.median([x['bootstrap_overlap_mean'] for x in rows])),'median_matrix_random_orientation_reference':float(np.median([x['random_orientation_expected_overlap'] for x in rows])),'matrices_with_any_bootstrap_rank_below2':sum(x['rank_below2_count']>0 for x in rows),'minimum_adjacent_relative_singular_gap':min(x['minimum_adjacent_relative_singular_gap'] for x in rows),'matrices_with_gap_below1e-3':sum(x['adjacent_relative_singular_gap_below1e-3_count']>0 for x in rows)}
report={'status':'FIT_ONLY_POSTHOC_DIAGNOSTIC_COMPLETE','plan':reg,'summary':summary,'rows':rows,'seconds':time.time()-t,'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.run/'fit-coefficient-gradients.pt',a.run/'protocol.json',a.bank]},'limitations':['Bootstrap uses only18 existing semantic examples; no additional validation information.','Stable estimated space does not establish semantic identification or downstream gain.','Diagonal singular-mode bank is not rotation-invariant under exact repeated singular values.','All original pilot predictions, primary contrast and settings unchanged.']}
(a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary))
