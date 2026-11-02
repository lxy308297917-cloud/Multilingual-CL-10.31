"""Cross-model proof from actual manifests and completed full predictions."""
import argparse,json
from pathlib import Path
from common import dump,jsonl,sha256,require_complete,object_hash

EXPECTED={'flores200_en_sw_full_devtest_v2':1012,'flores200_sw_en_full_devtest_v2':1012,'xlsum_sw_v2':987,'xlsum_en_v2':11535,'humaneval_plus_v2':164,'mbpp_plus_v2':378}
METHODS={'base','full','lora','ssu'}

def verify_group(group,expected):
 if set(group)!=METHODS:raise ValueError('Four methods required')
 baseline=group['base']
 for method,m in group.items():
  if m['method']!=method or m['scope']!='full' or len(m['sample_ids'])!=expected:raise ValueError('Wrong method/scope/cardinality')
  for field in ['benchmark_id','sample_ids','sample_ids_sha256','data_manifest_hash','tokenizer','prompt_tokens_sha256','backend','batch_size','source_files_sha256']:
   if m[field]!=baseline[field]:raise ValueError('Cross-model drift: '+field)
  if m['benchmark_protocol']!=baseline['benchmark_protocol']:raise ValueError('Benchmark protocol drift')
  if m['benchmark_protocol']['enable_thinking'] is not False:raise ValueError('Thinking enabled')
  if m['batch_size']!=32:raise ValueError('Fixed batch size drift')
  identity=m['checkpoint_identity'];has_adapter=any(Path(k).name=='adapter_model.safetensors' for k in identity)
  if has_adapter!=(method=='lora'):raise ValueError('Adapter identity mismatch')
  if not any(Path(k).name.startswith('model') and k.endswith('.safetensors') for k in identity):raise ValueError('Full base weights not hashed')
  if method!='base' and m['training_seed']!=42:raise ValueError('Wrong training seed')
 return True


def main():
 p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-pending',action='store_true');a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 state=json.loads(a.state.read_text());proof=[];pending=[]
 for task,expected in EXPECTED.items():
  group={};data={};sources=[]
  for method in METHODS:
   s=state['states'][f'{method}|{task}|full']
   if s['status']!='COMPLETE':pending.append(f'{method}|{task}|full');continue
   run=Path(s['output']);m=json.loads((run/'manifest.json').read_text());group[method]=m;rows=require_complete(jsonl(run/'predictions.jsonl'),m['sample_ids']);summary=json.loads(Path(s['summary']).read_text())
   if len(rows)!=expected or summary['samples']!=expected or summary['predictions_sha256']!=sha256(run/'predictions.jsonl'):raise ValueError('Counts or score identity mismatch')
   data[method]=object_hash([(r['id'],r.get('source'),r.get('reference'),r['prompt_token_sha256']) for r in rows]);sources.append(dict(method=method,manifest_sha256=sha256(run/'manifest.json'),predictions_sha256=summary['predictions_sha256'],summary_sha256=sha256(Path(s['summary']))))
  if set(group)==METHODS:
   verify_group(group,expected)
   if len(set(data.values()))!=1:raise ValueError('Actual per-sample prompt/data drift')
   proof.append(dict(benchmark=task,samples_per_model=expected,actual_prompt_data_hash=data['base'],four_models_verified=True,evidence=sources))
 if pending and not a.allow_pending:raise ValueError('Incomplete formal generation: '+str(pending))
 a.output.mkdir(parents=True);dump(a.output/'verification.json',dict(full_generation_verified=not pending,groups=proof,pending=pending,code_execution_status='NOT_RUN; generation verification cannot establish execution scores'))
 print(json.dumps({'complete_groups':len(proof),'pending_full_runs':len(pending),'output':str(a.output)}))
if __name__=='__main__':main()
