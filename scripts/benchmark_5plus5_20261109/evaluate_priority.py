"""Resource-only max5 adapter around unchanged frozen scientific evaluators."""
from pathlib import Path
import hashlib,importlib.util,json,sys,types
R=Path('/path/to/project')
def main():
 family=sys.argv.pop(1);assert family in ['fa','fft'];policy=json.loads((R/'configs/benchmark_5plus5_20261109/priority_resource_policy.json').read_text());record=policy['evaluators'][family];source=Path(record['path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==record['sha256']
 sys.path.insert(0,str(Path(__file__).parent));from gpu_slot_five import run_job as run_five
 def run_priority(build_command,*args,**kw):
  def command(uuid):
   cmd=list(map(str,build_command(uuid)))
   if '--max-user-gpus' in cmd:
    index=cmd.index('--max-user-gpus')+1;assert cmd[index]=='4';cmd[index]='5'
   return cmd
  return run_five(command,*args,**kw)
 module=types.ModuleType('gpu_slot');module.run_job=run_priority;sys.modules['gpu_slot']=module
 spec=importlib.util.spec_from_file_location('frozen_scientific_evaluator',source);evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator);evaluator.main()
if __name__=='__main__':main()
