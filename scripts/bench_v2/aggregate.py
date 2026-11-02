"""Build derived tables from verified full runs; missing results remain null."""
import argparse, csv, json, sys, datetime
from pathlib import Path
from common import jsonl, require_complete, sha256, dump
from paired_statistics import paired_corpus, paired_macro


def main():
 p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=1000);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 state=json.loads(a.state.read_text());a.output.mkdir(parents=True);rows=[];ready={};evidence=[]
 for task in state['tasks']:
  if task['scope']!='full':continue
  method,benchmark=task['method'],task['task'];s=state['states'][task['key']]
  item=dict(method=method,benchmark=benchmark,status=('FAILED' if s['status']=='FAILED' else 'PARTIAL' if s['status'] in ['RUNNING','SCORING','WAITING_SCORER'] else 'NOT_RUN'),execution_state=s['status'],samples=None,metrics=None)
  if s['status']=='COMPLETE':
   summary_path=Path(s['summary']);summary=json.loads(summary_path.read_text());run=Path(s['output']);manifest=json.loads((run/'manifest.json').read_text());preds=require_complete(jsonl(run/'predictions.jsonl'),manifest['sample_ids'])
   if summary['predictions_sha256']!=sha256(run/'predictions.jsonl'):raise ValueError('Prediction hash changed after scoring: '+str(run))
   if summary['samples']!=len(preds):raise ValueError('Scoring sample count differs')
   cost=json.loads((run/'cost.json').read_text()); cost['gpu_peak_allocated_bytes']=None; cost['torch_cuda_reserved_bytes']=None
   item.update(status=summary['run_status'],samples=summary['samples'],metrics=summary.get('metrics'),summary=str(summary_path),diagnostics=summary.get('diagnostics',summary.get('static_diagnostics')),checkpoint=manifest['checkpoint_identity'],direction=manifest['benchmark_protocol']['language'],protocol_version=manifest['result_schema_version'],generation_parameters=dict(backend=manifest['backend'],decoding=manifest['benchmark_protocol']['decoding_config'],batch_size=manifest['batch_size']),prediction_sha256=summary['predictions_sha256'],errors=summary.get('errors',0),missing=0,compute_cost=cost)
   ready[(benchmark,method)]=(preds,summary,summary_path)
   evidence.append(dict(key=task['key'],manifest_sha256=sha256(run/'manifest.json'),prediction_sha256=sha256(run/'predictions.jsonl'),summary_sha256=sha256(summary_path)))
  rows.append(item)
 comparisons=[]
 for (benchmark,method),(preds,summary,summary_path) in ready.items():
  if method=='base' or (benchmark,'base') not in ready or not benchmark.startswith(('flores','xlsum')):continue
  baseline,_,basepath=ready[(benchmark,'base')]
  if [r['id'] for r in preds]!=[r['id'] for r in baseline] or [r['reference'] for r in preds]!=[r['reference'] for r in baseline]:raise ValueError('Unpaired data')
  h=[r['prediction'] for r in preds];b=[r['prediction'] for r in baseline];refs=[r['reference'] for r in preds]
  metrics=['chrF++','BLEU'] if benchmark.startswith('flores') else ['chrF++']
  for metric in metrics:comparisons.append(dict(benchmark=benchmark,method=method,metric=metric,**paired_corpus(h,b,refs,metric,a.repeats)))
  if benchmark.startswith('xlsum'):
   target=json.loads((summary_path.parent/'rouge_per_sample.json').read_text());base=json.loads((basepath.parent/'rouge_per_sample.json').read_text())
   if [x['id'] for x in target]!=[x['id'] for x in base]:raise ValueError('Unpaired ROUGE IDs')
   for metric in ['ROUGE-1','ROUGE-2','ROUGE-Lsum']:comparisons.append(dict(benchmark=benchmark,method=method,metric=metric,**paired_macro([x[metric] for x in target],[x[metric] for x in base],a.repeats)))
 payload=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),state_sha256=sha256(a.state),analysis_source_sha256={name:sha256(Path(__file__).parent/name) for name in ['aggregate.py','paired_statistics.py']},results=rows,paired_comparisons=comparisons,evidence=evidence,limitations=['Code execution pass@1 remains NOT_RUN; static generation is not functional correctness.','Unadjusted exploratory intervals; no model selection from these formal test results.','Preliminary tables can contain pending tasks and do not establish final comparisons.'])
 dump(a.output/'results.json',payload)
 with (a.output/'results.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['method','benchmark','status','samples','metric','value','checkpoint','direction','protocol_version','generation_parameters','prediction_sha256','errors','missing','compute_cost']);writer.writeheader()
  for r in rows:
   for metric,value in (r['metrics'] or {'NOT_AVAILABLE':None}).items():writer.writerow(dict(method=r['method'],benchmark=r['benchmark'],status=r['status'],samples=r['samples'],metric=metric,value=value,**{k:json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in ['checkpoint','direction','protocol_version','generation_parameters','prediction_sha256','errors','missing','compute_cost']}))
 print(json.dumps({'rows':len(rows),'paired_comparisons':len(comparisons),'output':str(a.output)}))
if __name__=='__main__':main()
