"""Prepare pinned official test data without executing remote loading scripts."""
import argparse,json,tarfile,csv,hashlib,datetime
from pathlib import Path
from common import sha256,dump,object_hash,ids_hash,select_ids

REV_XL='30fece425f9a3866e04321773ca7a80056d55ca6'
REV_CROSS='b3fb19e9c135397c65f11fc9cf4ce58bf86aa29b'
LANG={'en':'English','sw':'Swahili'}
SUM_PROMPTS={
 'en':'Summarize the following article in English. Preserve the main facts, do not add information, and output only a concise summary.\n\n{article}',
 'sw':'Fupisha makala ifuatayo kwa Kiswahili. Hifadhi mambo makuu ya ukweli, usiongeze habari zisizokuwamo, na toa muhtasari mfupi tu.\n\n{article}',
}

def archive_test(path,expected_name):
 rows=[]
 with tarfile.open(path,'r|bz2') as archive:
  for member in archive:
   if Path(member.name).name==expected_name:
    if not member.isfile():raise ValueError('Dataset member is not a regular file')
    rows=[json.loads(line) for line in archive.extractfile(member) if line.strip()];break
 if not rows:raise ValueError('Test member missing/empty')
 return rows

def write_dataset(output,name,rows,source,metrics,max_new_tokens,prompt_template,legacy_ids=None):
 ids=[str(r['id']) for r in rows]
 if len(set(ids))!=len(ids) or not rows:raise ValueError('Duplicate/missing official IDs')
 directory=output/name;directory.mkdir();path=directory/'test.jsonl'
 with path.open('x') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 manifest=dict(benchmark_id=name,benchmark_version='2.1',source_repository=source['repository'],dataset_revision=source['revision'],dataset_split='test' if 'flores' not in name else 'devtest',sample_ids_sha256=ids_hash(rows),samples=len(rows),data_sha256=sha256(path),source=source,language=rows[0]['direction'],prompt_template=prompt_template,tokenizer_revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',chat_template='Base tokenizer.apply_chat_template(add_generation_prompt=True, enable_thinking=False)',enable_thinking=False,decoding_config=dict(temperature=0,top_p=1,top_k=-1,max_new_tokens=max_new_tokens,stop_token_ids=[151645,151643],seed=42),max_input_tokens=7168,max_model_len=8192,overlong_policy='Head truncate article tokens only before applying chat wrapper; count and retain every article' if 'xlsum' in name or 'crosssum' in name else 'Reject overlong prompt, no silent truncation',scorer_version={'sacrebleu':'2.6.0','rouge-score':'0.1.2'},metric_names=metrics,result_schema_version='bench-v2.1',run_status='NOT_RUN',pilot_ids=select_ids(rows,'bench-v2-pilot42',128 if 'sum' in name else 16),smoke_ids=select_ids(rows,'bench-v2-smoke42',4))
 if legacy_ids is not None:
  manifest['legacy_500_indices_zero_based']=legacy_ids;manifest['supplementary_indices_zero_based']=[i for i in range(1012) if i not in set(legacy_ids)];manifest['independent_of_legacy']=False
 dump(directory/'manifest.json',manifest);return manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--flores',type=Path,required=True);p.add_argument('--legacy-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 lock=json.loads(a.legacy_config.read_text());old=lock['devtest_evaluation_indices_zero_based'];shots=lock['dev_example_indices_zero_based'];assert len(old)==len(set(old))==500 and len(shots)==3
 dev={k:(a.flores/'dev'/f'{k}.dev').read_text().splitlines() for k in ['eng_Latn','swh_Latn']};test={k:(a.flores/'devtest'/f'{k}.devtest').read_text().splitlines() for k in dev};assert all(len(v)==997 for v in dev.values()) and all(len(v)==1012 for v in test.values())
 for lang in dev:
  if {dev[lang][i] for i in shots}&set(test[lang]):raise ValueError('Few-shot dev text overlaps devtest')
 summary={}
 for src,tgt,srck,tgtk in [('en','sw','eng_Latn','swh_Latn'),('sw','en','swh_Latn','eng_Latn')]:
  template=f'Translate the following {LANG[src]} text into {LANG[tgt]}. Output only the translation.\n\n'+'{text}'
  examples=[]
  for i in shots:examples.extend([dict(role='user',content=template.format(text=dev[srck][i])),dict(role='assistant',content=dev[tgtk][i])])
  rows=[dict(id=str(i),benchmark='flores200',split='devtest',direction=f'{src}->{tgt}',source=test[srck][i],reference=test[tgtk][i],messages=examples+[dict(role='user',content=template.format(text=test[srck][i]))],legacy500=i in set(old)) for i in range(1012)]
  name=f'flores200_{src}_{tgt}_full_devtest_v2';source=dict(repository='facebookresearch/flores',revision='official-200-tar-sha256:'+lock['source_tar_sha256'],split='devtest',dev_shot_indices=shots,dev_source_sha256=sha256(a.flores/'dev'/f'{srck}.dev'),dev_target_sha256=sha256(a.flores/'dev'/f'{tgtk}.dev'),test_source_sha256=sha256(a.flores/'devtest'/f'{srck}.devtest'),test_target_sha256=sha256(a.flores/'devtest'/f'{tgtk}.devtest'))
  summary[name]=write_dataset(a.output,name,rows,source,['chrF++','BLEU'],512,template,old)
 for lang,fullname in [('sw','swahili'),('en','english')]:
  path=a.raw/f'xlsum-{fullname}_XLSum_v2.0.tar.bz2';raw=archive_test(path,fullname+'_test.jsonl');rows=[dict(id=str(r['id']),benchmark='xlsum',split='test',direction=f'{lang}->{lang}',source=r['text'],reference=r['summary'],url=r['url'],title=r['title'],article_prompt_template=SUM_PROMPTS[lang]) for r in raw];name=f'xlsum_{lang}_v2';source=dict(repository='csebuetnlp/xlsum',revision=REV_XL,archive_sha256=sha256(path),dataset_version='2.0.0',not_parallel_to_other_language=True)
  summary[name]=write_dataset(a.output,name,rows,source,['ROUGE-1','ROUGE-2','ROUGE-Lsum','chrF++'],256,SUM_PROMPTS[lang])
 for srclang,tgtlang,src,tgt in [('en','sw','english','swahili'),('sw','en','swahili','english')]:
  path=a.raw/f'CrossSum-{src}-{tgt}_CrossSum.tar.bz2';raw=archive_test(path,f'{src}-{tgt}_test.jsonl');rows=[dict(id=hashlib.sha256((r['source_url']+'|'+r['target_url']).encode()).hexdigest(),benchmark='crosssum',split='test',direction=f'{srclang}->{tgtlang}',source=r['text'],reference=r['summary'],source_url=r['source_url'],target_url=r['target_url'],article_prompt_template=SUM_PROMPTS[tgtlang],quality='Official retrieval alignment; independent semantic validation pending') for r in raw];name=f'crosssum_{srclang}_{tgtlang}_v2';source=dict(repository='csebuetnlp/CrossSum',revision=REV_CROSS,archive_sha256=sha256(path),dataset_version='1.0.0',semantic_alignment_verified=False,shares_sources_with_xlsum=True)
  summary[name]=write_dataset(a.output,name,rows,source,['ROUGE-1','ROUGE-2','ROUGE-Lsum','chrF++'],256,SUM_PROMPTS[tgtlang])
 dump(a.output/'manifest.json',summary);print(json.dumps({k:v['samples'] for k,v in summary.items()}))
if __name__=='__main__':main()
