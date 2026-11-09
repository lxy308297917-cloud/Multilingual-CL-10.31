"""CPU-only finite XQuAD reporting, no inference launches or retries."""
from pathlib import Path
import datetime,json,subprocess,time
R=Path('/path/to/project');L=Path('/path/to/logs/qwen3-multilingual/xquad-20261109');P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
def main():
 deadline=time.monotonic()+120*3600
 while time.monotonic()<deadline:
  p=subprocess.run([P,str(Path(__file__).with_name('report.py'))],cwd=R,capture_output=True,text=True)
  with (L/'report-observer.jsonl').open('a') as f:f.write(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))+'\n')
  if p.returncode:raise RuntimeError('XQuAD raw audit failed; preserve outputs and inspect')
  if len(list(L.glob('*/DONE.json')))==10:return
  time.sleep(300)
 raise TimeoutError('Finite observer expired; inference jobs unchanged')
if __name__=='__main__':main()
