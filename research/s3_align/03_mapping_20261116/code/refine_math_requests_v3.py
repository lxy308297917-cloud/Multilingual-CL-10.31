"""Before any diagnostic outcomes, derive each prompt from upstream requests."""
from common import *
import random,shutil,pyarrow as pa,pyarrow.parquet as pq,inspect
from transformers import AutoTokenizer
from lm_eval.tasks import TaskManager,get_task_dict
import lm_eval
def main():
 assert not (R/'assets/diagnostic_states').exists() and not (LOG/'math256').exists()
 archive=R/'records/math_prefix_preexecution_v2';archive.mkdir(exist_ok=False)
 for p in [R/'inputs/math256.jsonl',R/'inputs/math256.parquet',R/'records/MATH_DATA_FROZEN.json',R/'DIAGNOSTIC_LOCK.json',R/'records/math_actual_framework_requests.jsonl']:shutil.copyfile(p,archive/p.name)
 plan=read(R/'records/MATH_DATA_FROZEN.json');source=Path(plan['source']);data=pq.read_table(source).to_pylist();lookup={r['question']:i for i,r in enumerate(data)};assert len(lookup)==7473
 histpath=Path(read(R/'records/HISTORICAL_MATH_DONE.json')['sources']['Base']['files'][0]['path']);historical=[r for r in rows(histpath) if r['filter']=='flexible-extract'];assert len(historical)==1319
 fewshot=set();occurrences=0
 for row in historical:
  prompt=row['arguments']['gen_args_0']['arg_0'];parts=prompt.split('Question: ')[1:];assert len(parts)==6
  questions=[p.split('\nAnswer:',1)[0] for p in parts];assert questions[-1]==row['doc']['question']
  for q in questions[:-1]:assert q in lookup;fewshot.add(lookup[q]);occurrences+=1
 excluded=set(plan['excluded_ids'])|fewshot;eligible=[i for i in range(7473) if i not in excluded];assert len(eligible)>=256
 selected=random.Random(20261116).sample(eligible,256);pq.write_table(pa.Table.from_pylist([data[i] for i in selected]),R/'inputs/math256.parquet')
 manager=TaskManager(include_path=str(R/'inputs/math_task'),include_defaults=False);task=get_task_dict(['s3map_math256'],task_manager=manager)['s3map_math256'];task.set_fewshot_seed(42)
 tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
 def chat(history,add_generation_prompt=True):return tok.apply_chat_template(history,tokenize=False,add_generation_prompt=add_generation_prompt,continue_final_message=not add_generation_prompt,enable_thinking=False)
 # Harness CLI automatically sets this True with --apply_chat_template.
 task.build_all_requests(rank=0,world_size=1,apply_chat_template=True,fewshot_as_multiturn=True,chat_template=chat,tokenizer_name=str(BASE).replace('/','__'))
 requests=sorted(task.instances,key=lambda x:x.doc_id);assert len(requests)==256
 result=[];actual=[];new_fewshot=set()
 for request,i in zip(requests,selected):
  d=data[i];assert request.doc==d;prefix=request.args[0];parts=prefix.split('Question: ')[1:];assert len(parts)==6
  context_ids=[lookup[p.split('\nAnswer:',1)[0]] for p in parts[:-1]];assert i not in context_ids;new_fewshot.update(context_ids)
  # Actual fewshots coincide with the source protocol's corresponding request index.
  old=historical[request.doc_id];expected=old['arguments']['gen_args_0']['arg_0'].replace('Question: '+old['doc']['question']+'\nAnswer:','Question: '+d['question']+'\nAnswer:')
  assert prefix==expected,'Actual upstream fewshot/chat protocol differs from historical generation'
  encoded=tok(prefix+d['answer'],add_special_tokens=False,return_offsets_mapping=True);ids=encoded['input_ids'];prefix_ids=tok.encode(prefix,add_special_tokens=False)
  targets=[j for j,(s,e) in enumerate(encoded['offset_mapping']) if s>=len(prefix) and e>s];positions=sorted({targets[round((len(targets)-1)*k/3)]-1 for k in range(4)})
  assert len(positions)==4 and len(ids)<8192 and ids[:len(prefix_ids)]==prefix_ids
  result.append(dict(id=f'math-train-{i}',source_id=i,question=d['question'],answer=d['answer'],reference_final=d['answer'].split('####')[-1].strip(),split='new_training_side_diagnostic',input_ids=ids,prefix=prefix,positions=positions,target_ids=[ids[p+1] for p in positions],prompt_tokens=len(prefix_ids),answer_tokens=len(targets),reference_steps=len(d['answer'].splitlines()),fewshot_source_ids=context_ids))
  actual.append(dict(doc_id=request.doc_id,source_id=i,prefix=prefix,kwargs=request.args[1],doc=request.doc,fewshot_source_ids=context_ids))
 assert not set(selected)&new_fewshot and new_fewshot<=fewshot
 assert not (R/'assets/diagnostic_states').exists() and not (LOG/'math256').exists()
 jsonl(R/'inputs/math256.jsonl',result);jsonl(R/'records/math_actual_framework_requests.jsonl',actual)
 plan.update(at=now(),revision='v3_actual_upstream_requests_before_any_diagnostic_outcome',excluded_ids=sorted(excluded),excluded_fewshot_ids=sorted(fewshot),selected_ids=selected,eligible_count=len(eligible),historical_fewshot_occurrences=occurrences,historical_fewshot_distinct=len(fewshot),new_fewshot_distinct=len(new_fewshot),fewshot_as_multiturn=True,teacher_forcing='Each exact upstream5shot request,including per-question sampled examples and multiturn chat template, followed by common referenceanswer;4fixedrule prediction positions',preexecution_revision_reason='Historical prompts vary across questions. Earlier draft reused firstprompt examples, and exclusion onlycovered firstprompt. Before any diagnostic output, expand exclusion to all1319 historical5shot contexts, resample256 withsamepredeclaredseed, construct actual upstream prompts. MainSFT andformalprotocol unchanged.',limitation='Excludes all known local functional pools,prior screening candidates,exact SFTquestion matches,andallhistoricalGSMfewshotdemonstrations. Still not evidence of no pretrained exposure.',prompt_tokens_range=[min(r['prompt_tokens'] for r in result),max(r['prompt_tokens'] for r in result)])
 plan['hashes']={p:sha(p) for p in plan['hashes']};write(R/'records/MATH_DATA_FROZEN.json',plan)
 lock=read(R/'DIAGNOSTIC_LOCK.json');lock['files'][str(R/'inputs/math256.jsonl')]=sha(R/'inputs/math256.jsonl');lock['files'][str(R/'inputs/math256.parquet')]=sha(R/'inputs/math256.parquet');lock['note']='v3 preexecution:256IDsresampled afterexcludingallhistoricalfewshots; everyprefixbuiltbyupstream requests andmatched to historical correspondingindex; no newdiagnostic predictions inspected';write(R/'DIAGNOSTIC_LOCK.json',lock)
 write(R/'records/MATH_REQUESTS_VERIFIED.json',dict(at=now(),n=256,each_prefix_matches_upstream_and_historical_corresponding_index=True,selected_ids_absent_from_all_current_and_historical_fewshot_demonstrations=True,historical_fewshot_unique=len(fewshot),excluded_total=len(excluded),eligible=len(eligible),max_full_tokens=max(len(r['input_ids']) for r in result),fewshot_as_multiturn=True,framework_source=str(Path(lm_eval.__file__).resolve()),framework_task_builder_sha256=sha(inspect.getfile(type(task))),cpu_only_no_scoring=True))
 print(json.dumps(read(R/'records/MATH_REQUESTS_VERIFIED.json')))
if __name__=='__main__':main()
