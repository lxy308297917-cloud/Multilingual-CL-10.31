"""CPU-only finite report refresh; never starts/retries inference or training."""
from pathlib import Path
import csv,datetime,json,subprocess,time
R=Path('/path/to/project');L=Path('/path/to/logs/qwen3-multilingual/benchmark-5plus5-20261109');P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
def main():
 deadline=time.monotonic()+120*3600
 while time.monotonic()<deadline:
  p=subprocess.run([P,str(Path(__file__).with_name('report.py'))],cwd=R,capture_output=True,text=True)
  with (L/'report-observer.jsonl').open('a') as f:f.write(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))+'\n')
  if p.returncode:raise RuntimeError('CPU report failed; inference jobs untouched; inspect report-observer.jsonl')
  count=len(list(L.glob('*/sib200/DONE.json')));gm=len(list(L.glob('*_historical_lora/gmmlu/DONE.json')))
  deferred=set(json.loads((R/'configs/benchmark_5plus5_20261109/deferred_models.json').read_text()))
  expected=27-len(deferred)
  rows=list(csv.DictReader((R/'research/benchmark_5plus5_20261109/scores_5plus5.csv').open()))
  eligible=[x for x in rows if x['model'] not in deferred]
  columns=['Belebele','XNLI','XL-Sum','Global-MMLU','SIB-200','GSM8K','IFEval','HumanEval+','EN MMLU','EN Belebele']
  matrix_complete=len(eligible)==expected and all(x[t]!='' for x in eligible for t in columns)
  if matrix_complete and count==expected and gm==2 and len(list(L.glob('*_historical_lora/code/DONE.json')))+len(list(L.glob('*_historical_lora/mmlu/DONE.json')))==5:return
  time.sleep(300)
 raise TimeoutError('Finite observer expired; jobs untouched')
if __name__=='__main__':main()
