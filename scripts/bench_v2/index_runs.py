"""Create an analysis index from explicitly selected completed runs; no queue needed."""
import argparse,datetime,json
from pathlib import Path
from common import jsonl,require_complete,sha256,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 tasks=[];states={}
 for run in a.run:
  m=json.loads((run/'manifest.json').read_text());done=json.loads((run/'DONE.json').read_text());key='|'.join([m['method'],m['benchmark_id'],m['scope']])
  if key in states:raise ValueError('Ambiguous duplicate selected runs: '+key)
  rows=require_complete(jsonl(run/'predictions.jsonl'),m['sample_ids']);digest=sha256(run/'predictions.jsonl')
  if done['predictions_sha256']!=digest:raise ValueError('Completion marker hash mismatch')
  candidates=[f for f in run.glob('*/summary.json') if f.parent.name.startswith(('score-','static-'))]
  if len(candidates)!=1:raise ValueError('Select exactly one scored result directory per run: '+str(run))
  summary=candidates[0];s=json.loads(summary.read_text())
  if s['samples']!=len(rows) or s['predictions_sha256']!=digest:raise ValueError('Scoring identity mismatch')
  if m['scope']=='full' and s['run_status'] not in ['VERIFIED_COMPLETE','NOT_RUN']:raise ValueError('Full score marked incomplete')
  tasks.append(dict(key=key,method=m['method'],task=m['benchmark_id'],scope=m['scope']));states[key]=dict(status='COMPLETE',run_status=s['run_status'],samples=len(rows),output=str(run.resolve()),summary=str(summary.resolve()))
 dump(a.output,dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),tasks=tasks,states=states,source='Explicit completed run selection; no private dispatcher state required'))
 print(json.dumps({'indexed_runs':len(tasks),'output':str(a.output)}))
if __name__=='__main__':main()
