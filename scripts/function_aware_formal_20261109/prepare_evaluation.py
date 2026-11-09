"""Official AR/RU Global-MMLU configurations, pinned to the existing SW revision."""
from pathlib import Path
import hashlib,json,re,shutil
import pyarrow.parquet as pq
import yaml,lm_eval
from huggingface_hub import hf_hub_download
R=Path('/path/to/project')
O=Path('/path/to/storage/datasets/function-aware-formal-20261109/evaluation-v1')
REV='0e619dbeb34206cd48705a1a0ea7fb21cae09993'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=False);source=Path(lm_eval.__file__).parent/'tasks/global_mmlu/full';records={}
 for lang in ['ar','ru']:
  data=O/'official-global-mmlu'/lang;data.mkdir(parents=True)
  files={}
  for split,count in [('dev',285),('test',14042)]:
   local=Path(hf_hub_download('CohereLabs/Global-MMLU',f'{lang}/{split}-00000-of-00001.parquet',repo_type='dataset',revision=REV,local_dir=O/'official-global-mmlu'))
   table=pq.read_table(local);assert table.num_rows==count
   assert len(set(table['subject'].to_pylist()))==57
   files[split]=dict(path=str(local),sha256=sha(local),rows=count)
  target=O/'tasks'/('global_mmlu_full_'+lang);target.mkdir(parents=True)
  prefix='global_mmlu_full_'+lang;group=prefix+'_locked';template='_'+lang+'_template_yaml'
  base=yaml.safe_load((source/lang/template).read_text());base['dataset_path']='parquet';base.pop('dataset_name');base['dataset_kwargs']={'data_files':{k:v['path'] for k,v in files.items()}}
  (target/template).write_text(yaml.safe_dump(base,sort_keys=False,allow_unicode=True));shutil.copy2(source/lang/'utils.py',target/'utils.py')
  names=[]
  for file in sorted((source/lang).glob(prefix+'_*.yaml')):
   name=file.stem+'_locked';text=file.read_text();text,n=re.subn(r'(?m)^"?task"?: .*$',f'task: {name}',text);assert n==1
   for category in ['stem','other','social_sciences','humanities']:text=text.replace(prefix+'_'+category+'_tasks',group+'_'+category+'_tasks')
   (target/(name+'.yaml')).write_text(text);names.append(name)
  assert len(names)==57
  (target/('_'+group+'.yaml')).write_text('group: '+group+'\ntask:\n'+''.join('  - '+n+'\n' for n in names)+'aggregate_metric_list:\n  - metric: acc\n    weight_by_size: true\n')
  records[lang]={'official_source':'https://huggingface.co/datasets/CohereLabs/Global-MMLU','revision':REV,'files':files,'subjects':57,'test_rows':14042,'fewshot':5,'batch_size':1,'prompt_and_scorer':'installed upstream official language task; data loader only changed to locally pinned parquet'}
 manifest=dict(status='FROZEN_OFFICIAL_KNOWLEDGE_TASKS_READY',languages=records,files={str(p.relative_to(O)):sha(p) for p in (O/'tasks').rglob('*') if p.is_file()},formal_test_used_to_set_weights=False)
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest['languages'],indent=2))
if __name__=='__main__':main()
