"""Official code extraction and non-executing AST diagnostics. Never exec/eval."""
import argparse,ast,json,hashlib,importlib.metadata as metadata
from pathlib import Path
from common import dump,jsonl,require_complete,sha256

def signature_accepts(tree,entrypoint,arity):
 definitions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==entrypoint]
 if not definitions:return False
 args=definitions[0].args;pos=len(args.posonlyargs)+len(args.args);minimum=pos-len(args.defaults)
 return minimum<=arity and (args.vararg is not None or arity<=pos) and not any(v is None for v in args.kw_defaults)

def analyze(row,sanitize,arities):
 raw=row['prediction'];code=sanitize(raw,entrypoint=row['entry_point']);syntax_error=None;signature=False
 try:tree=ast.parse(code);signature=bool(code.strip()) and all(signature_accepts(tree,row['entry_point'],a) for a in arities)
 except (SyntaxError,ValueError,RecursionError) as e:syntax_error=type(e).__name__
 return dict(id=row['id'],entry_point=row['entry_point'],sanitized_code=code,sanitized_code_sha256=hashlib.sha256(code.encode()).hexdigest(),empty_code=not code.strip(),syntax_error=syntax_error,signature_arity_valid=signature,contains_code_fence='```' in raw,hit_cap=row['generation_hit_cap'],generated_tokens=row['generated_tokens'],execution_status='NOT_RUN',runtime_error=None,timeout=None,pass_base=None,pass_plus=None)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--official-data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 if metadata.version('evalplus')!='0.3.1':raise ValueError('Use pinned EvalPlus0.3.1 extraction')
 from evalplus.sanitize import sanitize
 manifest=json.loads((a.run/'manifest.json').read_text());rows=require_complete(jsonl(a.run/'predictions.jsonl'),manifest['sample_ids']);problems={r['task_id']:r for r in jsonl(a.official_data)}
 expected=manifest['benchmark_protocol']['source']['archive_uncompressed_sha256']
 if sha256(a.official_data)!=expected:raise ValueError('Official code data identity differs')
 out=[analyze(r,sanitize,sorted({len(inputs) for inputs in problems[r['id']]['base_input']})) for r in rows]
 with (a.output/'sanitized_samples.jsonl').open('x') as f:
  for r in out:f.write(json.dumps(dict(task_id=r['id'],solution=r['sanitized_code']),ensure_ascii=False)+'\n')
 dump(a.output/'static_per_sample.json',out);summary=dict(benchmark=manifest['benchmark_id'],method=manifest['method'],scope=manifest['scope'],generation_status='VERIFIED_COMPLETE' if manifest['scope']=='full' else 'PILOT_ONLY',run_status='NOT_RUN',execution_status='NOT_RUN',reason='No confirmed strong execution sandbox; host execution forbidden. Docker permission denied; requested user-scope cgroup pids/memory/CPU limits did not take effect.',samples=len(out),metrics={'pass@1_base':None,'pass@1_plus':None},static_diagnostics={'empty_code_count':sum(r['empty_code'] for r in out),'syntax_error_count':sum(r['syntax_error'] is not None for r in out),'signature_arity_valid_count':sum(r['signature_arity_valid'] for r in out),'cap_count':sum(r['hit_cap'] for r in out),'fenced_output_count':sum(r['contains_code_fence'] for r in out),'runtime_error_count':None,'timeout_count':None},extractor_version='evalplus0.3.1 sanitize',signature_scope='Top-level Python function accepts each official base-input positional argument count; static heuristic, not execution correctness',predictions_sha256=sha256(a.run/'predictions.jsonl'),sanitized_samples_sha256=sha256(a.output/'sanitized_samples.jsonl'),official_dataset_sha256=sha256(a.official_data))
 dump(a.output/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
