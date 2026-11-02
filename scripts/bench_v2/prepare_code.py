"""Official EvalPlus data, default instruction prefix, Qwen3 nonthinking wrapper."""
import argparse,json,importlib.metadata as metadata
from pathlib import Path
from prepare_datasets import write_dataset
from common import sha256,dump

PREFIX='Please provide a self-contained Python script that solves the following problem in a markdown code block:'

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();bundle=json.loads((a.output/'manifest.json').read_text());result={}
 if metadata.version('evalplus')!='0.3.1':raise ValueError('EvalPlus version differs')
 for name,file,version,count in [('humaneval_plus_v2','HumanEvalPlus-v0.1.10.jsonl','v0.1.10',164),('mbpp_plus_v2','MbppPlus-v0.2.0.jsonl','v0.2.0',378)]:
  path=a.raw/file
  with path.open() as stream:problems=[json.loads(line) for line in stream]
  if len(problems)!=count or len({r['task_id'] for r in problems})!=count:raise ValueError('Wrong official full dataset cardinality')
  rows=[dict(id=r['task_id'],benchmark=name,split='official_full',direction='en->python',source=r['prompt'],reference=None,entry_point=r['entry_point'],messages=[dict(role='user',content=PREFIX+'\n```\n'+r['prompt'].strip()+'\n```\n')]) for r in problems]
  source=dict(repository='evalplus/'+('humanevalplus_release' if name.startswith('human') else 'mbppplus_release'),revision=version,archive_uncompressed_sha256=sha256(path),evalplus_version='0.3.1',data_scope='Official complete EvalPlus curated evaluation release; original MBPP base and plus are evaluated on the same378 tasks',prompt_compatibility='Official default user instruction prefix; native response-prefill replaced by Base Qwen3 apply_chat_template enable_thinking=False, one greedy complete Python script')
  config=write_dataset(a.output,name,rows,source,['pass@1_base','pass@1_plus'],1024,PREFIX+'\n```\n{official_problem_prompt}\n```\n');config['dataset_split']='official_full';config['scorer_version']={'evalplus':'0.3.1'};config['execution_status']='NOT_RUN';config['execution_gate']='Requires independently verified strong isolated executor; no host execution';dump(a.output/name/'manifest.json',config);bundle[name]=config;result[name]=count
 dump(a.output/'manifest.json',bundle);print(json.dumps(result))
if __name__=='__main__':main()
