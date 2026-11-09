"""Recover an authorized failed stage with frozen science and independent output.

The original failed attempt is archived separately. A logical output symlink
keeps existing report readers and the post-benchmark barrier compatible.
"""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,importlib.util,json,os,sys
R=Path('/path/to/project');L=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109')
REGISTRY=L/'recoveries.json'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def update(key,status,**more):
 with (L/'recovery-registry.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);registry=load(REGISTRY);registry['stages'][key].update(status=status,updated_at=datetime.datetime.now().astimezone().isoformat(),**more)
  temp=REGISTRY.with_suffix('.tmp');temp.write_text(json.dumps(registry,indent=2)+'\n');temp.replace(REGISTRY)
  # Registry contains provenance and paths, not GPU UUIDs or process inventory.
  (R/'research/function_aware_formal_20261109/recoveries.json').write_text(json.dumps(registry,indent=2)+'\n')
def owners(pid):
 result=[]
 for _ in range(8):
  p=Path('/proc')/str(pid)
  try:
   uid=p.stat().st_uid;fields=dict(x.split(':',1) for x in (p/'status').read_text().splitlines() if ':' in x);parent=int(fields['PPid']);result.append(dict(pid=pid,uid=uid,ppid=parent))
  except (OSError,KeyError,ValueError):break
  if uid!=os.getuid() or parent<=1:break
  pid=parent
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);a=ap.parse_args();record=load(REGISTRY)['stages'][a.key]
 language,method,stage=record['language'],record['method'],record['stage'];source=R/'scripts/function_aware_formal_20261109/evaluate.py'
 assert sha(source)==record['frozen_evaluation_source_sha256']==load(R/'research/function_aware_formal_20261109/freeze.json')['source_sha256']['evaluate.py']
 logical=Path(record['logical_output']);recovery=Path(record['recovery_output']);archive=Path(record['failed_archive'])
 assert logical.is_symlink() and logical.resolve(strict=False)==recovery.resolve(strict=False) and not recovery.exists()
 assert load(archive/'state.json')['status']=='FAILED' and sha(archive/'state.json')==record['failed_state_sha256']
 checkpoint=Path(record['checkpoint']);done=load(checkpoint/'TRAINING_DONE.json')
 assert done['steps']==400 and done['exposure']['samples']==6400 and sha(checkpoint/'adapter_model.safetensors')==record['model_sha256']==done['weights_sha256']['adapter_model.safetensors']
 dataset=Path(record['dataset']);manifest=load(dataset/'manifest.json');assert manifest['samples']==987 and sha(dataset/'test.jsonl')==record['dataset_data_sha256']==manifest['data_sha256'] and sha(dataset/'manifest.json')==record['dataset_manifest_sha256']
 update(a.key,'RECOVERY_QUEUED_OR_RUNNING')
 sys.path.insert(0,str(R/'scripts/aya_4b_multilang_20261106'));import gpu_slot
 original_run=gpu_slot.run_job;original_snapshot=gpu_slot.queue.snapshot;selected={}
 def snapshot():
  state=original_snapshot();uuid=selected.get('uuid')
  if uuid and uuid in state:
   card=state[uuid]
   children=Path('/proc')/str(os.getpid())/'task'/str(os.getpid())/'children'
   try:direct=[int(x) for x in children.read_text().split()]
   except OSError:direct=[]
   event=dict(at=datetime.datetime.now().astimezone().isoformat(),gpu_uuid=uuid,launch_parent_pid=os.getpid(),direct_children=direct,gpu_pids=card['pids'],memory_mib=card['memory_mib'],process_ancestry=[owners(pid) for pid in card['pids']])
   with (Path(record['recovery_base'])/'handoff-trace.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
  return state
 def run(builder,env,log,save,**kwargs):
  def traced(status,**values):
   if status=='LAUNCHING':selected['uuid']=values['gpu_uuid']
   save(status,**values)
  gpu_slot.queue.snapshot=snapshot
  try:return original_run(builder,env,log,traced,**kwargs)
  finally:gpu_slot.queue.snapshot=original_snapshot
 gpu_slot.run_job=run
 spec=importlib.util.spec_from_file_location('frozen_formal_evaluation_recovery',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.L=Path(record['recovery_base']);sys.argv=[str(source),'--language',language,'--method',method,'--stage',stage]
 try:
  module.main();marker=load(recovery/'DONE.json');assert marker['status']=='FIXED_ENDPOINT_STAGE_COMPLETE_PENDING_PAIRED_AUDIT'
  assert marker['model_sha256']==record['model_sha256'] and marker['no_checkpoint_selection']
  assert sha(source)==record['frozen_evaluation_source_sha256'];update(a.key,'RECOVERY_COMPLETE_PENDING_RAW_AUDIT',recovered_done_sha256=sha(recovery/'DONE.json'))
 except Exception as exc:update(a.key,'RECOVERY_FAILED',recovery_error=repr(exc));raise
if __name__=='__main__':main()
