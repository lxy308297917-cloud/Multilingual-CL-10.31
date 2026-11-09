"""Freeze official XQuAD AR/RU validation data and standard generation tasks."""
import collections,datetime,hashlib,inspect,json,shutil
from pathlib import Path
import transformers.data.metrics.squad_metrics as metrics
R=Path('/path/to/project');D=Path('/path/to/storage/datasets/xquad-supplement-20261109/v1');C=R/'configs/xquad_20261109'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (D/'manifest.json').exists();(D/'tasks').mkdir();shutil.copy2(Path(__file__).with_name('utils.py'),D/'tasks/utils.py')
 rev=json.loads((D/'source/revision.json').read_text());manifest=dict(**rev,split='official XQuAD evaluation; translated subset of SQuAD validation',languages={},shots=0,nonthinking=True,greedy=True,max_new_tokens=128,stop=['\n'],metrics=['official SQuAD v1.1 exact_match','official SQuAD v1.1 token_f1'],normalization='Official English SQuAD v1.1 lower/punctuation/articles/whitespace rules, including for AR/RU',multiple_gold='maximum over all references, as official evaluate-v1.1; unlike installed Harness XQuAD first-reference-only helper',scorer_source_sha256=sha(Path(inspect.getfile(metrics))),harness_inference='lm_eval generate_until hf; language-native built-in XQuAD prompt',sw='UNSUPPORTED: no synthetic SW translation or substitute dataset')
 prompts={'ar':'سيا: {{context}}\n\nسؤال: {{question}}\n\nإجابة:','ru':'Контекст: {{context}}\n\nВопрос: {{question}}\n\nОтвет:'};allids=[]
 import yaml
 for lang in ['ar','ru']:
  source=D/'source'/('xquad.'+lang+'.json');raw=json.loads(source.read_text());items=[];paragraphs=0;mismatches=[]
  for article in raw['data']:
   for para in article['paragraphs']:
    paragraphs+=1
    for q in para['qas']:
     assert q['answers'];answers=dict(text=[a['text'] for a in q['answers']],answer_start=[a['answer_start'] for a in q['answers']]);items.append(dict(id=q['id'],title=article['title'],context=para['context'],question=q['question'],answers=answers,paragraph_sha256=hashlib.sha256(para['context'].encode()).hexdigest()))
     for a in q['answers']:
      if para['context'][a['answer_start']:a['answer_start']+len(a['text'])]!=a['text']:mismatches.append(q['id'])
  assert len(items)==1190 and paragraphs==240 and len({x['id'] for x in items})==1190;allids.append([x['id'] for x in items]);data=D/(lang+'.jsonl');data.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items))
  task=dict(task='mcl_xquad_'+lang,dataset_path='json',dataset_kwargs=dict(data_files=dict(validation=str(data))),validation_split='validation',output_type='generate_until',doc_to_text=prompts[lang],doc_to_target='{{answers["text"][0]}}',target_delimiter=' ',num_fewshot=0,generation_kwargs=dict(until=['\n'],do_sample=False,temperature=0.,max_gen_toks=128),metric_list=[dict(metric=k,aggregation='mean',higher_is_better=True) for k in ['exact_match','f1']],metadata=dict(version=1));p=D/'tasks'/('xquad_'+lang+'.yaml');p.write_text(yaml.safe_dump(task,sort_keys=False,allow_unicode=True)+'process_results: !function utils.process_results\n')
  manifest['languages'][lang]=dict(samples=1190,paragraphs=240,source_sha256=sha(source),jsonl_sha256=sha(data),task_sha256=sha(p),prompt=prompts[lang],answer_offset_mismatch_ids=mismatches,offset_note='Offsets audited only; scoring uses all official reference strings; no examples repaired or removed',reference_count_distribution=dict(collections.Counter(len(x['answers']['text']) for x in items)))
 assert allids[0]==allids[1];manifest['created_at']=datetime.datetime.now().astimezone().isoformat();(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(C/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:{x:v[x] for x in ['samples','paragraphs','reference_count_distribution']} for k,v in manifest['languages'].items()}))
if __name__=='__main__':main()
