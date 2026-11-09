"""Fill only missing historical General5 scores with the frozen original engines."""
from pathlib import Path
import json,datetime,subprocess,time,os
R=Path('/path/to/project');L=Path('/path/to/logs/qwen3-multilingual/benchmark-5plus5-20261109');P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
def main():
 todo=[(lang+'_historical_lora','code') for lang in ['sw','ar','ru']]+[(lang+'_historical_lora','mmlu') for lang in ['ar','ru']];failed=[]
 for name,stage in todo[:]:
  (L/'general-schedule.json').write_text(json.dumps(dict(status='RUNNING_OR_WAITING_GPU',at=datetime.datetime.now().astimezone().isoformat(),pid=os.getpid(),pending=todo,failed=failed),indent=2)+'\n')
  if (L/name/stage/'DONE.json').exists():todo.remove((name,stage));continue
  if (L/name/stage).exists():raise RuntimeError('Existing unfinished general benchmark attempt; preserve and diagnose')
  p=subprocess.run([P,str(Path(__file__).with_name('evaluate_general.py')),'--model',name,'--stage',stage],cwd=R)
  if p.returncode:failed.append(dict(model=name,stage=stage,exit_code=p.returncode))
  todo.remove((name,stage))
 (L/'general-schedule.json').write_text(json.dumps(dict(status='ALL_TERMINAL_PENDING_RAW_AUDIT',at=datetime.datetime.now().astimezone().isoformat(),pending=todo,failed=failed),indent=2)+'\n')
if __name__=='__main__':main()
