"""Finite shared-server GPU handoff; account actual jobs plus live queue reservations."""
import os,sys,time,json,subprocess,fcntl,datetime,signal
from pathlib import Path
sys.path.insert(0,'/path/to/user/gpu-queue')
import gpu_queue as queue
queue.select_queue('aya4b-multilang-observer')
R=Path('/path/to/project');PY='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
def descendant(pid,root):
 for _ in range(64):
  if pid==root:return True
  try:info=dict(line.split(':',1) for line in Path('/proc/'+str(pid)+'/status').read_text().splitlines() if ':' in line);pid=int(info['PPid'])
  except (OSError,KeyError,ValueError):return False
  if pid<=1:return False
 return False

def run_job(build_command,env,log,save,deadline_hours=24):
 deadline=time.monotonic()+deadline_hours*3600;save('WAITING_FREE_GPU_ACCOUNT_AND_QUEUE_MAX5');child=None
 while child is None:
  if time.monotonic()>deadline:raise TimeoutError('Finite24h GPU wait expired')
  with (queue.BASE_ROOT/'reservation.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);snapshot=queue.snapshot();reserved=queue.other_queue_gpus();assigned=queue.user_gpu_assignments(snapshot,reserved)
   for uuid,gpu in snapshot.items():
    if len(assigned|{uuid})>5 or uuid in reserved or gpu['pids'] or gpu['memory_mib']>1000 or gpu['util']>5:continue
    probe=subprocess.run([PY,str(R/'scripts/preflight_gpu.py'),'--uuid',uuid,'--max-user-gpus','5','--settle-seconds','5'],capture_output=True,text=True)
    if probe.returncode:continue
    command=list(map(str,build_command(uuid)));environment={**env,'CUDA_VISIBLE_DEVICES':uuid,'AYA_EXPECTED_GPU_UUID':uuid};log.parent.mkdir(parents=True,exist_ok=True);(log.parent/(log.name+'.command.json')).write_text(json.dumps({'command':command,'gpu_uuid':uuid,'account_gpu_limit':5,'queue_reserved_uuids':sorted(reserved),'started_at':datetime.datetime.now().astimezone().isoformat(),'gpu_preflight':json.loads(probe.stdout)},indent=2)+'\n');save('LAUNCHING',gpu_uuid=uuid,command=command)
    stream=log.open('x');child=subprocess.Popen(command,cwd=R,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);handoff=time.monotonic()+180
    try:
     while child.poll() is None:
      state=queue.snapshot()[uuid];compute=[pid for pid in state['pids'] if descendant(pid,child.pid)]
      if compute:break
      if state['pids']:raise RuntimeError('GPU acquired by another job before this child; cancel only this own child group')
      if time.monotonic()>handoff:raise TimeoutError('No CUDA process after180s handoff; preserve this attempt for inspection')
      time.sleep(2)
    except Exception:
     if child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
     stream.close();raise
    save('RUNNING',gpu_uuid=uuid,child_pid=child.pid);break
  if child is None:time.sleep(60)
 code=child.wait();stream.close()
 if code:raise subprocess.CalledProcessError(code,command)
 return uuid
