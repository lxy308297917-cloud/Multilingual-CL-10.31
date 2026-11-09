"""Finite read-only FFT reporting; never manages or retries GPU jobs."""
from pathlib import Path
import json,subprocess,time
R=Path('/path/to/project')
P='/path/to/storage/envs/qwen3-multilingual-20261031/bin/python'
L=Path('/path/to/logs/qwen3-multilingual/fft-representation-20261109')
def main():
 deadline=time.monotonic()+120*3600
 while time.monotonic()<deadline:
  with (L/'report-observer.log').open('a') as stream:
   subprocess.run([P,str(Path(__file__).with_name('collect_results.py'))],cwd=R,stdout=stream,stderr=subprocess.STDOUT,check=True)
  report=json.loads((R/'research/fft_representation_20261109/progress.json').read_text())
  terminal={'COMPLETE_FIXED_FINAL400','FAILED'}
  schedule=report['evaluation']
  if all(x['status'] in terminal for x in report['training'].values()) and not schedule.get('pending') and not schedule.get('active'):return
  time.sleep(300)
 raise TimeoutError('Finite reporting deadline; GPU tasks untouched')
if __name__=='__main__':main()
