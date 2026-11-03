"""Prepare existing verified SUM/MT aggregate input with true SST method identity."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.project
assert not a.output.exists();done=json.loads((a.record/'DONE.json').read_text());assert done['actual_method'] in ['sst_norefresh','sst_r10'] and done['exit_code']==0
launch=json.loads((a.record/'launch.json').read_text());assert launch['actual_method']==done['actual_method'] and not launch['formal_test_used_for_selection']
old=json.loads((r/'results/bench_v2/summary/results.json').read_text());tasks=[];states={};evidence=[]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
for entry in launch['commands']:
 bench=entry['task']
 if not bench.startswith(('xlsum','flores')):continue
 cmd=entry['command'];target=Path(cmd[cmd.index('--output')+1]);ts=target.parent/'scores/summary.json'
 baseidentity=next(x for x in old['results'] if x['method']=='base' and x['benchmark']==bench and x['status']=='VERIFIED_COMPLETE')
 candidates=[x for x in (r/'results/bench_v2').glob('base-'+bench+'-full-*/score-v1/summary.json') if json.loads(x.read_text())['predictions_sha256']==baseidentity['prediction_sha256']]
 assert len(candidates)==1;c=candidates[0];base=c.parent.parent
 for method,folder,summary in [('base',base,c),(done['actual_method'],target,ts)]:
  d=json.loads(summary.read_text());assert d['run_status']=='VERIFIED_COMPLETE' and d['scope']=='full' and sha(folder/'predictions.jsonl')==d['predictions_sha256']
  key=bench+'|'+method;tasks.append({'key':key,'task':bench,'method':method,'scope':'full'});states[key]={'status':'COMPLETE','summary':str(summary),'output':str(folder)}
  evidence.append({'key':key,'summary_sha256':sha(summary),'prediction_sha256':sha(folder/'predictions.jsonl'),'manifest_sha256':sha(folder/'manifest.json')})
 def rows(path):return [json.loads(x) for x in path.open()]
 before,after=rows(base/'predictions.jsonl'),rows(target/'predictions.jsonl');assert len(before)==len(after)==baseidentity['samples']
 for b,t in zip(before,after):
  for key in ['id','reference','prompt','prompt_token_sha256']:assert b[key]==t[key],(bench,key,b['id'])
 for field in ['data_sha256','result_schema_version']:
  bm=json.loads((base/'manifest.json').read_text());tm=json.loads((target/'manifest.json').read_text())
  if field in bm or field in tm:assert bm.get(field)==tm.get(field),(bench,field)
state={'tasks':tasks,'states':states,'true_method':done['actual_method'],'legacy_manifest_artifact_kind':'full','kind_is_not_FFT_scientific_identity':True,'wrapper_launch_sha256':sha(a.record/'launch.json'),'wrapper_done_sha256':sha(a.record/'DONE.json'),'aligned_doc_prompt_and_token_hashes':True,'evidence':evidence,'formal_test_used_for_selection':False}
a.output.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps({'tasks':len(tasks),'prompt_hashes_aligned':True,'true_method':done['actual_method']}))
