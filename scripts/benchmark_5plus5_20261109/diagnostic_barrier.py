"""Keep auxiliary diagnostics behind the newly authorized formal benchmarks."""
from pathlib import Path
import datetime,json,os,subprocess,time
R=Path('/path/to/project');L=Path('/path/to/logs/qwen3-multilingual/benchmark-5plus5-20261109');P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
def main():
 deadline=time.monotonic()+120*3600
 while time.monotonic()<deadline:
  sib=len(list(L.glob('*/sib200/DONE.json')));gm=len(list(L.glob('*_historical_lora/gmmlu/DONE.json')))
  (L/'diagnostic-barrier.json').write_text(json.dumps(dict(status='WAITING_NEW_FORMAL_BENCHMARKS_NO_GPU_RESERVED',at=datetime.datetime.now().astimezone().isoformat(),sib_completed=sib,sib_required=27,historical_gm_completed=gm,historical_gm_required=2,pid=os.getpid()),indent=2)+'\n')
  if sib==27 and gm==2:
   # Original script still enforces all113 original formal stages itself.
   subprocess.run([P,str(R/'scripts/function_aware_formal_20261109/diagnose_final.py')],cwd=R,check=True);return
  time.sleep(60)
 raise TimeoutError('Finite diagnostic barrier expired; no GPU started')
if __name__=='__main__':main()
