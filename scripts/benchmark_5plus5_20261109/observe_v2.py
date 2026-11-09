"""CPU-only finite report refresh; never starts/retries inference or training."""
from pathlib import Path
import datetime,json,subprocess,time
R=Path('/path/to/project');L=Path('/path/to/logs/qwen3-multilingual/benchmark-5plus5-20261109');P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
def main():
 deadline=time.monotonic()+120*3600
 while time.monotonic()<deadline:
  p=subprocess.run([P,str(Path(__file__).with_name('report.py'))],cwd=R,capture_output=True,text=True)
  with (L/'report-observer.jsonl').open('a') as f:f.write(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))+'\n')
  if p.returncode:raise RuntimeError('CPU report failed; inference jobs untouched; inspect report-observer.jsonl')
  count=len(list(L.glob('*/sib200/DONE.json')));gm=len(list(L.glob('*_historical_lora/gmmlu/DONE.json')))
  if count==27-len(json.loads((R/'configs/benchmark_5plus5_20261109/deferred_models.json').read_text())) and gm==2:return
  time.sleep(300)
 raise TimeoutError('Finite observer expired; jobs untouched')
if __name__=='__main__':main()
