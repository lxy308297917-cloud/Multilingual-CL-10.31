"""Recompute all mechanism summaries from frozen features; no model selection."""
from pathlib import Path
import csv,hashlib,json,re,sys
import numpy as np
import torch
from safetensors.torch import load_file
from metrics import covariance_difference,normalized_drift,cosine
R=Path('/data0/users/lixiaoyang/10.31/multilingual-research')
D=Path('/data1/lixiaoyang/datasets/representation-benchmarks-20261109/v1')
N=R/'research/representation_benchmarks_20261109'
MODELS=['Base','all_layer','full_feature','ce_replay']
CONDITIONS=['raw_content_mean','chat_all_tokens_mean','chat_content_tokens_mean']
def csvwrite(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);out=N/'mechanism';out.mkdir(parents=True,exist_ok=False)
 root=D/'features';data={};sources={}
 for m in MODELS:
  p=root/m;d=json.loads((p/'DONE.json').read_text());assert d['status']=='COMPLETE' and d['old_rows']==256 and d['pairs']==300
  for name,digest in d['hashes'].items():assert sha(p/name)==digest
  data[m]=dict(old=load_file(str(p/'old.safetensors'))['features'],rows=json.loads((p/'old_rows.json').read_text()),parallel=load_file(str(p/'parallel.safetensors'))['pooled'])
  sources[m]=dict(done_sha256=sha(p/'DONE.json'),hashes=d['hashes'],seconds=d['seconds'],peak_allocated=d['gpu_peak_allocated_bytes'])
 reference=data['Base'];assert all(data[m]['rows']==reference['rows'] for m in MODELS)
 oldcache=D.parents[1]/'functional-representation-20261109/p1-v1/base_features_v1'
 prior=load_file(str(oldcache/'base_features.safetensors'))['features'];assert prior.shape==reference['old'].shape
 cache_check=float(normalized_drift(prior,reference['old']).max());assert cache_check<1e-8,cache_check
 drift_items=[];drifts=[];covs=[];literal=[];pooled={}
 for m in MODELS:
  pooled[m]=[]
  for row in reference['rows']:
   start=row['feature_offset_start'];slots=[start+i for i in row['answer_prediction_slots']]
   x=reference['old'][slots];y=data[m]['old'][slots];d=normalized_drift(x,y);pooled[m].append(y.float().mean(0))
   for l,v in enumerate(d.tolist(),1):drift_items.append(dict(model=m,id=row['id'],split=row['split'],capability=row['capability'],layer=l,normalized_squared_l2=v))
  pooled[m]=torch.stack(pooled[m])
 for split in ['build','check']:
  for cap in ['math','instruction']:
   idx=[i for i,x in enumerate(reference['rows']) if x['split']==split and x['capability']==cap];assert len(idx)==64
   for m in MODELS:
    for l in range(1,37):
     vals=np.array([x['normalized_squared_l2'] for x in drift_items if x['model']==m and x['split']==split and x['capability']==cap and x['layer']==l])
     assert len(vals)==64
     drifts.append(dict(model=m,split=split,capability=cap,layer=l,n=64,mean=float(vals.mean()),median=float(np.median(vals)),std=float(vals.std(ddof=0))))
     covs.append(dict(model=m,split=split,capability=cap,layer=l,**covariance_difference(pooled['Base'][idx,l-1],pooled[m][idx,l-1])))
    # Literal cross-layer formula: average examples then center36 layer rows.
    literal.append(dict(model=m,split=split,capability=cap,definition='Paper literal cross-layer covariance, selected-answer-token pooling adaptation; one matrix, not layer-wise',**covariance_difference(pooled['Base'][idx].mean(0),pooled[m][idx].mean(0))))
 csvwrite(out/'old_drift_items.csv',drift_items);csvwrite(out/'old_drift_layers.csv',drifts);csvwrite(out/'sample_covariance_layers.csv',covs);csvwrite(out/'paper_literal_cross_layer_covariance.csv',literal)
 sims=[];simitems=[];base_sim={};shuffled_sim={};permutation=torch.arange(300).roll(-1)
 for m in MODELS:
  p=data[m]['parallel'];assert p.shape==(300,2,3,37,2560)
  for k,c in enumerate(CONDITIONS):
   s=cosine(p[:,0,k],p[:,1,k]);sh=cosine(p[:,0,k],p[permutation,1,k]);base_sim[m,c]=s;shuffled_sim[m,c]=sh
 for m in MODELS:
  for c in CONDITIONS:
   s=base_sim[m,c];delta=s-base_sim['Base',c];sh=shuffled_sim[m,c]
   pairs=list(map(json.loads,(D/'parallel/pairs.jsonl').open()))
   for l in range(37):
    sims.append(dict(model=m,condition=c,layer=l,n=300,mean=float(s[:,l].mean()),std=float(s[:,l].std(correction=0)),delta_mean_vs_base=float(delta[:,l].mean()),delta_std_vs_base=float(delta[:,l].std(correction=0)),shuffled_mean=float(sh[:,l].mean()),matched_minus_shuffled_mean=float((s[:,l]-sh[:,l]).mean())))
    for i,row in enumerate(pairs):simitems.append(dict(model=m,condition=c,id=row['id'],layer=l,cosine=float(s[i,l]),delta_vs_base=float(delta[i,l]),shuffled_cosine=float(sh[i,l])))
 csvwrite(out/'cross_language_layers.csv',sims);csvwrite(out/'cross_language_items.csv',simitems)
 norms=[]
 assets=json.loads((R/'configs/representation_benchmarks_20261109/assets.json').read_text())
 for m in MODELS:
  total=0.
  if assets['models'][m]['adapter']:
   p=Path(assets['models'][m]['adapter']);cfg=json.loads((p/'adapter_config.json').read_text());state=load_file(str(p/'adapter_model.safetensors'))
   for name,a in state.items():
    if '.lora_A.' not in name:continue
    b=state[name.replace('.lora_A.','.lora_B.')];a=a.double();b=b.double();scale=cfg['lora_alpha']/cfg['r'];total+=float(((b.T@b)*(a@a.T).T).sum()*scale**2)
  norms.append(dict(model=m,effective_update_squared_frobenius=total,effective_update_frobenius=total**.5))
 csvwrite(out/'effective_update_norm.csv',norms)
 # Model-level descriptive summaries; the four models are NOT independent replicates.
 summary=[]
 for m in MODELS:
  row=dict(model=m,**{k:v for k,v in next(x for x in norms if x['model']==m).items() if k!='model'})
  for cap in ['math','instruction']:
   row['check_'+cap+'_l2_mean36']=float(np.mean([x['mean'] for x in drifts if x['model']==m and x['split']=='check' and x['capability']==cap]))
   row['check_'+cap+'_sample_cov_fro_mean36']=float(np.mean([x['normalized_frobenius'] for x in covs if x['model']==m and x['split']=='check' and x['capability']==cap]))
  for c in CONDITIONS:
   for label,lo,hi in [('early',1,12),('middle',13,24),('late',25,36)]:
    row[c+'_'+label+'_delta']=float(base_sim[m,c][:,lo:hi+1].sub(base_sim['Base',c][:,lo:hi+1]).mean())
  summary.append(row)
 csvwrite(out/'model_level_summary.csv',summary)
 # All reported curves use identical raw block/layer conventions.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':9,'savefig.dpi':180})
 colors=dict(zip(MODELS,['black','#2864b4','#168348','#b54b33']))
 for filename,table,key,ylabel in [('old_l2_check',drifts,'mean','Normalized squared L2'),('sample_covariance_check',covs,'normalized_frobenius','Normalized Frobenius covariance difference')]:
  fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharex=True)
  for ax,cap in zip(axes,['math','instruction']):
   for m in MODELS:
    rows=[x for x in table if x['model']==m and x['split']=='check' and x['capability']==cap];ax.plot([x['layer'] for x in rows],[x[key] for x in rows],label=m,color=colors[m])
   ax.set(title=cap+' check64',xlabel='Raw transformer block (1..36)',ylabel=ylabel);ax.grid(alpha=.2)
  axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(out/(filename+'.png'));plt.close(fig)
 for filename,key,ylabel in [('cross_language_cosine','mean','Mean EN-SW cosine'),('cross_language_delta','delta_mean_vs_base','Mean paired cosine difference vs Base'),('matched_minus_shuffled','matched_minus_shuffled_mean','Matched minus fixed shuffled cosine')]:
  fig,axes=plt.subplots(1,3,figsize=(13,3.6),sharex=True)
  for ax,c in zip(axes,CONDITIONS):
   for m in MODELS:
    rows=[x for x in sims if x['model']==m and x['condition']==c];ax.plot([x['layer'] for x in rows],[x[key] for x in rows],label=m,color=colors[m])
   ax.set(title=c,xlabel='0 embedding;1..36 raw blocks',ylabel=ylabel);ax.grid(alpha=.2)
  axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(out/(filename+'.png'));plt.close(fig)
 report=dict(status='COMPLETE',source_commit='50640bb',models=MODELS,fresh_base_vs_P1_cache_max_normalized_squared_l2=cache_check,primary_generalization='check128,64 math and64 instruction; build128 exposed in P2',sources=sources,zero_base_drift=all(x['mean']==0 for x in drifts if x['model']=='Base'),shuffled_control='One cyclic shift of frozen hash-selected pair order, fixed before feature extraction; no additional prompting or model generation',paper_layerwise_curve='NOT_IDENTIFIABLE_FROM_PAPER_FORMULA; do not label sample covariance as exact original implementation',new_training_jobs=0)
 (out/'DONE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
