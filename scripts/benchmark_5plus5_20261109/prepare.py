"""Prepare official test-only SIB data and document exact text overlap."""
import collections,csv,datetime,hashlib,json,unicodedata
from pathlib import Path
import yaml
R=Path('/path/to/project')
D=Path('/path/to/storage/datasets/benchmark-5plus5-20261109/v1')
LABELS=['science/technology','travel','politics','sports','health','entertainment','geography']
PROMPT='Classify the topic of the following text. Choose exactly one label from: '+', '.join(LABELS)+'.\n\nText: {{text}}\nTopic:'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def rows(p):return [json.loads(x) for x in p.open() if x.strip()]
def main():
 assert not (D/'manifest.json').exists();(D/'tasks').mkdir(exist_ok=False)
 manifest=dict(dataset='Davlan/sib200',revision='38977a667f6fc264d5c26ec57a01e16db040b358',split='test',labels=LABELS,prompt=PROMPT,shots=0,chat_template='Official Qwen3-4B apply_chat_template enable_thinking=False',primary='Harness acc_norm: sum loglikelihood divided by label Unicode character length, without target delimiter',secondary='raw sum loglikelihood acc; never select max of normalized/raw',candidate_ties='NumPy argmax first candidate',label_order_frozen=True,code_switched_topic_classification=True,not_original_author_best_prompt=True,languages={})
 overlap={};ids=[]
 for lang,config in [('sw','swh_Latn'),('ar','arb_Arab'),('ru','rus_Cyrl')]:
  source=D/'source'/(lang+'.tsv');raw=list(csv.DictReader(source.open(),delimiter='\t'));assert len(raw)==204 and len({x['index_id'] for x in raw})==204;assert set(x['category'] for x in raw)==set(LABELS)
  prepared=[dict(index_id=int(x['index_id']),text=x['text'],category=x['category'],label=LABELS.index(x['category'])) for x in raw];ids.append([(x['index_id'],x['category']) for x in prepared]);target=D/(lang+'.jsonl');target.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in prepared))
  task=dict(task='mcl5_sib200_'+lang,dataset_path='json',dataset_kwargs=dict(data_files=dict(test=str(target))),test_split='test',output_type='multiple_choice',doc_to_text=PROMPT,doc_to_choice=LABELS,doc_to_target='{{label}}',target_delimiter=' ',num_fewshot=0,metric_list=[dict(metric='acc_norm',aggregation='mean',higher_is_better=True),dict(metric='acc',aggregation='mean',higher_is_better=True)],metadata=dict(version=1))
  taskpath=D/'tasks'/('sib200_'+lang+'.yaml');taskpath.write_text(yaml.safe_dump(task,sort_keys=False,allow_unicode=True));counts=dict(collections.Counter(x['category'] for x in raw));manifest['languages'][lang]=dict(config=config,samples=204,source_sha256=sha(source),jsonl_sha256=sha(target),task_sha256=sha(taskpath),class_counts=counts,majority_baseline=max(counts.values())/204*100,sample_ids=[x['index_id'] for x in prepared])
  bp=Path('/path/to/storage/datasets/belebele-official-20261031/swh_Latn.jsonl') if lang=='sw' else Path('/path/to/storage/datasets/aya-4b-multilang-20261106/evaluation-v1')/(config+'.jsonl');br=rows(bp);passages={norm(x['flores_passage']) for x in br}
  tp=Path('/path/to/storage/datasets/aya-4b-20261105/mixture-v1/train6400.jsonl') if lang=='sw' else Path('/path/to/storage/datasets/aya-4b-multilang-20261106/mixtures')/lang/'train6400.jsonl';train=rows(tp);assert len(train)==6400;trainparts=[(i,norm(str(x.get('instruction',''))+'\n'+str(x.get('output','')))) for i,x in enumerate(train)]
  hits=[]
  for x in prepared:
   text=norm(x['text']);exact=sum(text==s for s in passages);contained=sum(text in s or s in text for s in passages if min(len(text),len(s))>=20);training=[i for i,s in trainparts if len(text)>=20 and text in s];hits.append(dict(index_id=x['index_id'],belebele_exact_passages=exact,belebele_containment_passages=contained,training_full_text_containment_rows=training))
  overlap[lang]=dict(method='NFKC casefold whitespace normalized equality and bidirectional full-passage containment; train uses full SIB text containment. No fuzzy/semantic dedup or post-test filtering.',belebele_source=str(bp),belebele_sha256=sha(bp),train_source=str(tp),train_sha256=sha(tp),sib_samples=204,belebele_unique_passages=len(passages),sib_exact_overlap=sum(x['belebele_exact_passages']>0 for x in hits),sib_containment_overlap=sum(x['belebele_containment_passages']>0 for x in hits),sib_train_fulltext_overlap=sum(bool(x['training_full_text_containment_rows']) for x in hits),items=hits,id_audit='SIB official index_id is retained; Belebele exposes source URL, question number and passage but no canonical FLORES integer ID. No unverified ID correspondence asserted.',limitations='Exact/containment checks do not exclude paraphrases, partial fragments or pretraining exposure. No test rows removed or tuned.')
 assert ids[0]==ids[1]==ids[2]
 manifest['created_at']=datetime.datetime.now().astimezone().isoformat();(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');Q=R/'research/benchmark_5plus5_20261109';(Q/'sib_overlap_audit.json').write_text(json.dumps(overlap,indent=2)+'\n');(R/'configs/benchmark_5plus5_20261109/sib_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({lang:{k:v for k,v in d.items() if k in ['sib_samples','belebele_unique_passages','sib_exact_overlap','sib_containment_overlap','sib_train_fulltext_overlap']} for lang,d in overlap.items()}))
if __name__=='__main__':main()
