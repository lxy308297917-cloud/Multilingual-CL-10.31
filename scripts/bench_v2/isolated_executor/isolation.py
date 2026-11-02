import os,signal,subprocess,time,resource
from pathlib import Path
A=Path(__file__).resolve().parents[2];E=Path('/path/to/code-eval-env');BASE=Path('/path/to/training-runtime-env')
def command(script,input_path=None):
 b=A/'work/code_executor_build/build/bwrap';runtime=A/'work/code_executor/runtime';helper=A/'work/code_executor_mbpp/kernel_limits.py'
 cmd=[str(b),'--as-pid-1','--unshare-all','--unshare-user','--uid',str(os.getuid()),'--gid',str(os.getuid()),'--clearenv','--cap-drop','ALL','--new-session','--die-with-parent','--disable-userns','--ro-bind',str(runtime),'/runtime','--ro-bind',str(BASE/'lib'),'/runtime/lib','--ro-bind',str(E/'lib/python3.11/site-packages'),'/runtime/lib/python3.11/site-packages','--ro-bind','/lib/x86_64-linux-gnu','/lib/x86_64-linux-gnu','--ro-bind','/usr/lib/x86_64-linux-gnu','/usr/lib/x86_64-linux-gnu','--ro-bind','/lib64','/lib64','--proc','/proc','--remount-ro','/proc','--dev','/dev','--size',str(16*1024**2),'--tmpfs','/dev/shm','--remount-ro','/dev','--size',str(64*1024**2),'--tmpfs','/work','--dir','/work/tmp','--dir','/work/home','--symlink','/work/tmp','/tmp','--chdir','/work','--ro-bind',str(script),'/entry.py','--ro-bind',str(helper),'/kernel_limits.py']
 if input_path is not None:cmd+=['--ro-bind',str(input_path),'/input.json']
 cmd+=['--remount-ro','/']
 env={'MALLOC_CONF':'background_thread:false','LD_LIBRARY_PATH':'/runtime/lib','PATH':'/runtime/bin','HOME':'/work/home','TMPDIR':'/work/tmp','PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','EVALPLUS_MAX_MEMORY_BYTES':str(4*1024**3)}
 for k,v in env.items():cmd+=['--setenv',k,v]
 return cmd+['/runtime/bin/python','/entry.py']
def outer_limits():
 for name,value in [('RLIMIT_AS',4*1024**3),('RLIMIT_DATA',4*1024**3),('RLIMIT_STACK',4*1024**3),('RLIMIT_CPU',110),('RLIMIT_FSIZE',8*1024**2),('RLIMIT_CORE',0)]:resource.setrlimit(getattr(resource,name),(value,value))
def run(script,input_path,stdout_path,stderr_path,wall=120):
 cmd=command(script,input_path);start=time.monotonic();timed_out=False
 with Path(stdout_path).open('xb') as out,Path(stderr_path).open('xb') as err:
  p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,preexec_fn=outer_limits)
  try:code=p.wait(timeout=wall)
  except subprocess.TimeoutExpired:
   timed_out=True;os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=5)
 return dict(exit_code=code,wall_timeout=timed_out,wall_seconds=time.monotonic()-start,outer_created_pid=p.pid,command=cmd)
