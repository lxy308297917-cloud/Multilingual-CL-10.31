"""Freeze existing four weights and supplementary benchmark inputs, no training."""
from pathlib import Path
import collections
import hashlib
import json
import shutil
import yaml

R=Path('/data0/users/lixiaoyang/10.31/multilingual-research')
D=Path('/data1/lixiaoyang/datasets/representation-benchmarks-20261109/v1')
K=Path('/data1/lixiaoyang/checkpoints/functional_representation_20261109/p2')
M=Path('/data0/users/lixiaoyang/models/Qwen3-4B')
C=R/'configs/representation_benchmarks_20261109'
REV='1cfa9a7208912126459214e8b04321603b3df60c'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
 assert not D.exists(),'Independent input directory required'
 D.mkdir(parents=True);C.mkdir(parents=True,exist_ok=True)
 lock=json.loads((R/'configs/aya_4b_20261105/model.lock.json').read_text())
 assert lock['repository']=='Qwen/Qwen3-4B' and lock['revision']==REV
 model_files={}
 for name,record in lock['files'].items():
  assert sha(M/name)==record['sha256'],name
  model_files[name]=record['sha256']
 models={'Base':dict(base_path=str(M),adapter=None,revision=REV)}
 for mode in ['all_layer','full_feature','ce_replay']:
  path=K/(mode+'_final_v1');done=json.loads((path/'TRAINING_DONE.json').read_text())
  assert done['steps']==400 and done['target_order_matches_original_all6400']
  assert done['exposure']['samples']==6400
  assert sha(path/'adapter_model.safetensors')==done['weights_sha256']['adapter_model.safetensors']
  models[mode]=dict(base_path=str(M),adapter=str(path.resolve()),revision=REV,
      adapter_sha256=sha(path/'adapter_model.safetensors'),adapter_config_sha256=sha(path/'adapter_config.json'),
      training_done_sha256=sha(path/'TRAINING_DONE.json'),fixed_step=400,target_samples=6400,training_microbatch=2)
 src=Path('/data1/lixiaoyang/datasets/aya-4b-20261105/supplemental-v1')
 old_manifest=json.loads((src/'manifest.json').read_text())
 xdir=D/'xnli';xdir.mkdir();tasks=D/'tasks';tasks.mkdir()
 ids=[str(i) for i in range(5010)]
 ranked=sorted(ids,key=lambda x:hashlib.sha256(('representation-benchmarks-v1-xnli-rotation|'+x).encode()).hexdigest())
 offsets={key:i%3 for i,key in enumerate(ranked)}
 assert collections.Counter(offsets.values())=={0:1670,1:1670,2:1670}
 task_records=[]
 for lang in ['sw','en']:
  source=src/('xnli_'+lang+'_test.jsonl');rows=list(map(json.loads,source.open()))
  expected=next(x for x in old_manifest['tasks'] if x['task']=='xnli_'+lang)['splits']['test']['sha256']
  assert sha(source)==expected and len(rows)==5010 and [x['id'] for x in rows]==ids
  validation=list(map(json.loads,(src/('xnli_'+lang+'_validation.jsonl')).open()))
  smoke_ids={str(x['id']) for label in range(3) for x in [r for r in validation if r['label']==label][:2]}
  assert len(smoke_ids)==6
  for scope,subset in [('formal',rows),('smoke',[x for x in validation if x['id'] in smoke_ids])]:
   for kind in ['primary','rotated']:
    selected=[]
    for x in subset:
     row=dict(x);offset=offsets[x['id']] if scope=='formal' else int(x['id'])%3
     row['original_label']=x['label'];row['rotation_offset']=offset if kind=='rotated' else 0
     row['semantic_at_letter']=[(i+row['rotation_offset'])%3 for i in range(3)]
     if kind=='rotated':
      suffix='A. Entailment\nB. Neutral\nC. Contradiction\nAnswer:'
      assert x['prompt'].endswith(suffix)
      names=['Entailment','Neutral','Contradiction']
      replacement='\n'.join(chr(65+i)+'. '+names[s] for i,s in enumerate(row['semantic_at_letter']))+'\nAnswer:'
      row['prompt']=x['prompt'][:-len(suffix)]+replacement
      row['label']=(x['label']-row['rotation_offset'])%3
     assert row['semantic_at_letter'][row['label']]==row['original_label']
     selected.append(row)
    target=xdir/(lang+'_'+scope+'_'+kind+'.jsonl')
    target.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in selected))
    name='rb_xnli_'+lang+'_'+scope+'_'+kind
    cfg=dict(task=name,dataset_path='json',dataset_kwargs={'data_files':{'test':str(target)}},test_split='test',
      output_type='multiple_choice',doc_to_text='{{prompt}}',doc_to_choice=list('ABC'),doc_to_target='label',
      num_fewshot=0,metric_list=[dict(metric='acc',aggregation='mean',higher_is_better=True)],metadata={'version':'1.0'})
    y=tasks/(name+'.yaml');y.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    task_records.append(dict(name=name,language=lang,scope=scope,kind=kind,samples=len(selected),
      data=str(target),data_sha256=sha(target),task_config_sha256=sha(y),
      gold_letter_counts=dict(collections.Counter(x['label'] for x in selected)),
      original_semantic_counts=dict(collections.Counter(x['original_label'] for x in selected))))
 old_code=Path('/data1/lixiaoyang/datasets/bench-v2-20261102/prepared-v1/humaneval_plus_v2')
 code=D/'code';code.mkdir();shutil.copy2(old_code/'test.jsonl',code/'test.jsonl')
 cfg=json.loads((old_code/'manifest.json').read_text());assert cfg['samples']==164 and sha(code/'test.jsonl')==cfg['data_sha256']
 cfg.update(tokenizer_revision=REV,model_revision=REV,benchmark_version='rb-20261109-v1',
     local_protocol_id='Qwen3-4B-nonthinking-greedy1024-V0',source_prepared_manifest_sha256=sha(old_code/'manifest.json'))
 cfg['source']=dict(cfg['source'],prompt_compatibility='Same official user instruction; Qwen3-4B non-thinking chat template, not old1.7B tokenizer')
 dump(code/'manifest.json',cfg)
 old=Path('/data1/lixiaoyang/datasets/functional-representation-20261109/p1-v1')
 rows=list(map(json.loads,(old/'encoded_selected.jsonl').open()));assert len(rows)==256
 assert collections.Counter((x['split'],x['capability']) for x in rows)=={(s,c):64 for s in ['build','check'] for c in ['math','instruction']}
 report=dict(status='BENCHMARK_INPUTS_FROZEN_BEFORE_NEW_SCORING',source_commit='50640bb271b646baf4e7cdec48648f8a8418fdfb',
     model_revision=REV,model_files=model_files,models=models,old_pool=dict(encoded_sha256=sha(old/'encoded_selected.jsonl'),
     rows=256,build_rows=128,check_rows=128,build_used_in_P2=True,check_used_in_P2=False),tasks=task_records,
     xnli=dict(source_manifest_sha256=sha(src/'manifest.json'),fewshot=0,batch_size=4,seed=42,
     candidate_likelihood='Original bare A/B/C candidates; no free-generation parser',
     rotation='Hash-sort IDs with fixed namespace then rank modulo3; exactly1670 offsets each, same map EN/SW; independent of labels/model scores',
     rotation_is_auxiliary_not_primary=True,bootstrap_unit='shared premise cluster',bootstrap_resamples=2000),
     code=dict(dataset_manifest=str(code/'manifest.json'),dataset_manifest_sha256=sha(code/'manifest.json'),
     official_release='HumanEvalPlus-v0.1.10',samples=164,greedy=True,samples_per_task=1,max_new_tokens=1024,
     vllm_engine='V0',batch_size=8,scorer='Existing official EvalPlus0.3.1 with documented find_zero bookkeeping fix; verified isolation required'),
     new_training_jobs=0,loss_changes=False,new_training_seeds=0,max_account_gpus=4,
     mechanism_status='Definitions, independent parallel corpus and extraction protocol being finalized; no mechanism result yet')
 dump(C/'assets.json',report);dump(D/'manifest.json',report)
 print(json.dumps(dict(models=list(models),xnli_tasks=len(task_records),formal_xnli_rows_per_model=20040,code_tasks=164,
    old_build=128,old_check=128,new_training_jobs=0)))

if __name__=='__main__':main()
