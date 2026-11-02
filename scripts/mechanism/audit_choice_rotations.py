"""Audit two option rotations using frozen predictions; source schema is required.
No model inference, formal-test selection, or primary-score updates.
"""
from pathlib import Path
import csv,json,collections,unicodedata
import argparse,hashlib
parser=argparse.ArgumentParser(description='Read-only post-hoc option rotation audit; no inference or score changes')
parser.add_argument('--run',action='append',required=True,help='Unique label=completed_module_swap_directory')
parser.add_argument('--probes',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();O=args.output;O.mkdir(parents=True,exist_ok=False)
probes={r['id']:r for r in map(json.loads,args.probes.open())}
runs=[]
for specification in args.run:
 label,separator,directory=specification.partition('=')
 if not separator or not label or not directory or any(r['label']==label for r in runs):raise ValueError('Require unique nonempty label=directory specifications')
 runs.append({'label':label,'stage':'intervention','output':directory})
rows=[];pairs=[]
norm=lambda s:' '.join(unicodedata.normalize('NFKC',s).casefold().split())
for run in [r for r in runs if r['stage']=='intervention']:
 directory=Path(run['output']);done=json.loads((directory/'DONE.json').read_text())
 for condition in done['conditions']:
  data=[r for r in map(json.loads,(directory/condition/'predictions.jsonl').open()) if r['task']=='knowledge_choice'];groups=collections.defaultdict(list)
  for r in data:groups[(r['input_language'],probes[r['id']]['pair_id'])].append(r)
  for (lang,pair),items in groups.items():
   assert len(items)==2;items.sort(key=lambda x:x['id']);source=[probes[x['id']] for x in items];gold=[norm(r['choices']['ABCD'.index(r['answer'])]) for r in source];assert gold[0]==gold[1];assert set(map(norm,source[0]['choices']))==set(map(norm,source[1]['choices']));assert len(set(map(norm,source[0]['choices'])))==4
   chosen=[norm(p['choices']['ABCD'.index(r['score']['predicted_option'])]) for r,p in zip(items,source)];correct=[bool(r['score']['content_correct']) for r in items];assert correct==[x==y for x,y in zip(chosen,gold)]
   pairs.append(dict(model=run['label'],condition=condition,input_language=lang,pair_id=pair,both_rotations_correct=all(correct),one_rotation_correct=sum(correct)==1,neither_rotation_correct=not any(correct),same_chosen_answer_content=chosen[0]==chosen[1],same_chosen_option_letter=items[0]['score']['predicted_option']==items[1]['score']['predicted_option']))
  for lang in ['en','sw']:
   subset=[x for x in pairs if x['model']==run['label'] and x['condition']==condition and x['input_language']==lang];assert len(subset)==22;preds=[r for r in data if r['input_language']==lang];dist=collections.Counter(r['score']['predicted_option'] for r in preds)
   rows.append(dict(model=run['label'],condition=condition,input_language=lang,predictions=44,source_pairs=22,original_correct_rows=sum(bool(r['score']['content_correct']) for r in preds),both_rotations_correct_pairs=sum(r['both_rotations_correct'] for r in subset),one_rotation_correct_pairs=sum(r['one_rotation_correct'] for r in subset),neither_rotation_correct_pairs=sum(r['neither_rotation_correct'] for r in subset),same_chosen_content_pairs=sum(r['same_chosen_answer_content'] for r in subset),same_option_letter_pairs=sum(r['same_chosen_option_letter'] for r in subset),**{f'predicted_{k}':dist[k] for k in 'ABCD'}))
for name,data in [('choice_rotation_summary.csv',rows),('choice_rotation_per_pair.csv',pairs)]:
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
report={'scope':'Post-hoc audit of already frozen development predictions only; not a new selection metric or formal benchmark','rows':len(rows),'paired_cases':len(pairs),'primary_scores_changed':False,'all_source_pairs_have_identical_gold_and_option_content_across_rotations':True,'all_original_correct_flags_recomputed_from_canonical_answer_content':True,'probes_sha256':hashlib.sha256(args.probes.read_bytes()).hexdigest(),'base_and_finals':[r for r in rows if r['condition'] in ['base','final']]};(O/'choice_rotation_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
