"""Verify complete task IDs, score MT/SUM, and retain supplementary MT subsets."""
import argparse,json,time,importlib.metadata as metadata
from pathlib import Path
import numpy as np
from common import jsonl,sha256,dump,require_complete
from scoring import mt_metrics,rouge_rows,output_diagnostics

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();manifest=json.loads((a.run/'manifest.json').read_text());expected=manifest['sample_ids'];rows=require_complete(jsonl(a.run/'predictions.jsonl'),expected);cfg=manifest['benchmark_protocol'];benchmark=cfg['benchmark_id'];pred=[r['prediction'] for r in rows];refs=[r['reference'] for r in rows]
 if benchmark.startswith('flores'):
  metrics,signature=mt_metrics(pred,refs);subsets={}
  for label,subset in [('legacy500_members_new_generation',[r for r in rows if r['legacy500']]),('supplementary512',[r for r in rows if not r['legacy500']])]:
   if subset:subsets[label]={'samples':len(subset),'metrics':mt_metrics([r['prediction'] for r in subset],[r['reference'] for r in subset])[0],'ids':[r['id'] for r in subset]}
  per_sample=None
 elif benchmark.startswith(('xlsum','crosssum')):
  per_sample=rouge_rows(pred,refs);metrics={k:float(np.mean([r[k] for r in per_sample])) for k in per_sample[0]};chrf,signature=mt_metrics(pred,refs);metrics['chrF++']=chrf['chrF++'];subsets={}
  dump(a.output/'rouge_per_sample.json',[dict(id=row['id'],**score) for row,score in zip(rows,per_sample)])
 else:raise ValueError('Use isolated EvalPlus scorer for code; pass@1 cannot be fabricated')
 diagnostics=output_diagnostics(rows);dump(a.output/'diagnostics_per_sample.json',diagnostics);status='VERIFIED_COMPLETE' if manifest['scope']=='full' else 'PILOT_ONLY';summary=dict(benchmark=benchmark,method=manifest['method'],run_status=status,scope=manifest['scope'],samples=len(rows),metrics=metrics,signatures=signature,metric_protocol={'sacrebleu':metadata.version('sacrebleu'),'rouge_score':metadata.version('rouge-score'),'rouge_stemmer':False,'rouge_tokenizer':'rouge_score default lower-case ASCII alphanumeric tokenizer','rouge_sentence_split':'NLTK English Punkt for both EN/SW; locked but not a validated SW sentence model'},subsets=subsets,source_manifest_sha256=sha256(a.run/'manifest.json'),predictions_sha256=sha256(a.run/'predictions.jsonl'),errors=0,missing=0,scoring_cpu_wall_seconds=time.monotonic()-start,diagnostics={'empty_count':sum(r['empty'] for r in diagnostics),'repetition_count':sum(r['repetitive'] for r in diagnostics),'cap_count':sum(r['hit_cap'] for r in diagnostics),'input_truncated_count':sum(r['article_truncated'] for r in diagnostics),'language_measured':sum(r['language_match'] is not None for r in diagnostics),'language_mismatch':sum(r['language_match'] is False for r in diagnostics),'language_unknown':sum(r['language_match'] is None for r in diagnostics)})
 dump(a.output/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
