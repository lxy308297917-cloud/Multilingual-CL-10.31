"""Fail-closed execution policy checks. This module never executes generated code.

A Python venv, timeout, or namespace availability alone is not sufficient proof.
"""
import argparse,json,os,shutil,subprocess,re
from common import dump


def validate_container_policy(policy):
 required={'network':'none','read_only_root':True,'privileged':False,'host_pid':False,'host_ipc':False,'no_new_privileges':True,'cap_drop_all':True,'devices':[],'sensitive_host_mounts':[],'limits_measured_in_kernel':True}
 for key,value in required.items():
  if policy.get(key)!=value:raise ValueError('Unsafe or unverified execution policy: '+key)
 for key,maximum in [('memory_bytes',4*1024**3),('pids_limit',64),('cpus',4),('per_task_wall_seconds',120)]:
  value=policy.get(key)
  if not isinstance(value,(int,float)) or isinstance(value,bool) or not 0<value<=maximum:raise ValueError('Missing/unbounded resource: '+key)
 if not isinstance(policy.get('uid'),int) or isinstance(policy['uid'],bool) or policy['uid']<=0:raise ValueError('Non-root UID required')
 if re.fullmatch(r'sha256:[0-9a-f]{64}',policy.get('image_digest','')) is None:raise ValueError('Pinned actual image digest required')
 return True


def require_execution_proof(proof):
 if proof.get('executor') not in ['verified_container','equivalent_verified_sandbox']:raise ValueError('A venv or process timeout is not a sandbox')
 if not proof.get('host_files_inaccessible') or not proof.get('network_block_test_passed') or not proof.get('resource_enforcement_test_passed'):raise ValueError('Missing functional isolation evidence')
 return validate_container_policy(proof['policy'])


def docker_info_accessible(returncode,stdout,stderr):
 # Older Docker CLI can return0 and JSON null on daemon permission errors.
 if returncode!=0 or stderr.strip():return False
 try:return isinstance(json.loads(stdout),list)
 except (ValueError,TypeError):return False


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();docker=shutil.which('docker');result={'execution_status':'NOT_RUN','generated_code_execution_allowed':False,'docker_binary':bool(docker),'bwrap_binary':bool(shutil.which('bwrap')),'note':'Detection is not execution authorization or isolation proof; no host code execution fallback'}
 if docker:
  q=subprocess.run([docker,'info','--format','{{json .SecurityOptions}}'],capture_output=True,text=True,timeout=15);result['docker_info_exit_code']=q.returncode;result['docker_daemon_accessible']=docker_info_accessible(q.returncode,q.stdout,q.stderr);result['docker_error']=q.stderr.strip()[:1000]
 result['reason']='No verified isolated executor configured; code generation/static diagnostics continue, pass@1 remains null/NOT_RUN';dump(a.output,result);print(json.dumps(result))
if __name__=='__main__':main()
