"""Preserve old scores and compare overlapping IDs without claiming independent tests."""
import argparse,json
from pathlib import Path
from common import dump,jsonl,sha256
from scoring import mt_metrics,repetition

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-index',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 state=json.loads(a.state.read_text());out=[]
 for entry in json.loads(a.legacy_index.read_text()):
  method=entry['method'];direction='en_sw' if entry['task'].startswith('eng') else 'sw_en';key=f'{method}|flores200_{direction}_full_devtest_v2|full';info=state['states'][key]
  if info['status']!='COMPLETE':continue
  old_summary_path=Path(entry['official_result_file']);summary=json.loads(old_summary_path.read_text())[entry['task']];oldpath=Path(summary['output']);old=jsonl(oldpath);new=jsonl(Path(info['output'])/'predictions.jsonl');byid={x['id']:x for x in new};matched=[]
  for x in old:
   y=byid[str(x['devtest_index_zero_based'])]
   if x['source']!=y['source'] or x['reference']!=y['reference']:raise ValueError('Legacy membership mismatch')
   matched.append(y)
  old_recomputed,signatures=mt_metrics([x['prediction'] for x in old],[x['reference'] for x in old])
  if abs(old_recomputed['chrF++']-summary['chrF++'])>1e-10:raise ValueError('Legacy score reproduction differs')
  new_metrics,_=mt_metrics([x['prediction'] for x in matched],[x['reference'] for x in matched])
  out.append(dict(method=method,direction=direction,legacy_samples=len(old),legacy_frozen_chrf=summary['chrF++'],legacy_recomputed=old_recomputed,legacy_prediction_sha256=sha256(oldpath),legacy_summary_sha256=sha256(old_summary_path),legacy_signatures=signatures,new_generation_on_same_500=new_metrics,identical_prediction_count=sum(x['prediction']==y['prediction'] for x,y in zip(old,matched)),legacy_repetition_count=sum(repetition(x['prediction']) for x in old),new_overlap_repetition_count=sum(repetition(x['prediction']) for x in matched),full_samples=len(new),supplementary_samples=len(new)-len(old),independent_of_legacy=False))
 a.output.mkdir(parents=True);dump(a.output/'legacy_comparison.json',dict(results=out,interpretation='Legacy scores remain frozen. New generation uses vLLM BF16 rather than historical Transformers BF16; identical greedy settings do not guarantee identical predictions. Overlapping500 and full1012 are not independent. Legacy BLEU recomputed here is an additional derived diagnostic, not a previously published score.'))
 print(json.dumps({'comparisons':len(out),'output':str(a.output)}))
if __name__=='__main__':main()
