from pathlib import Path
import json,gzip,hashlib,random
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
OLD=Path('/srv/research/reproductions/shared_subspace_analysis_20261115')
W=Path('/srv/research/Documents/Codex/2026-11-10/new-chat/work/s3-bactrian-20261113')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not (R/'protocol.json').exists()
 prior=json.loads((OLD/'protocol.json').read_text());tok=AutoTokenizer.from_pretrained(prior['models']['Base'],local_files_only=True)
 pool=[json.loads(x) for x in (W/'prepared/alignment.jsonl').read_text().splitlines()]
 raw={l:{r['id']:r for r in json.load(gzip.open(s['path'],'rt'))} for l,s in prior['sources'].items()}
 counts={l:0 for l in raw};examples=[]
 for r in pool:
  for l in raw:
   source=raw[l][r['id']];inp=source.get('input') or ''
   text=source['instruction'].strip()+('\n\n'+inp.strip() if inp else '')
   assert text==r['texts'][l]
   assert tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)==r['encoded'][l]['input_ids']
   counts[l]+=bool(inp.strip())
   if l=='en' and inp.strip() and len(examples)<3:examples.append(dict(id=r['id'],source=source,actual_text=text,body_positions=r['encoded'][l]['body_positions']))
 write(R/'records/input_definition_audit.json',dict(rows=1024,all_five_language_full_input_matches=True,nonempty_input=counts,examples=examples,correction='Actual historical inputs include instruction plus nonempty input; attachment instruction-only allegation is false. instruction-only in manifest means prompt without answer.'))
 rows=[dict(**r,split='fit') for r in pool]
 rows += [r for r in map(json.loads,(OLD/'inputs.jsonl').read_text().splitlines()) if r['split']=='dev']
 (R/'inputs/representation.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 sources={l:Path('/srv/storage/datasets/belebele-official-20261031')/(c+'.jsonl') for l,c in [('en','eng_Latn'),('sw','swh_Latn')]}
 data={l:[json.loads(x) for x in p.read_text().splitlines()] for l,p in sources.items()}
 sw={(d['link'],d['question_number']):d for d in data['sw']}
 order=list(range(900));random.Random(20261116).shuffle(order);selected=[];seen=set()
 for i in order:
  e=data['en'][i];s=sw[(e['link'],e['question_number'])]
  if e['flores_passage'] in seen:continue
  assert e['correct_answer_num']==s['correct_answer_num']
  seen.add(e['flores_passage']);selected.append(dict(id=i,en=e,sw=s,gold=int(e['correct_answer_num'])-1))
  if len(selected)==128:break
 for i,r in enumerate(selected):
  others=[j for j,s in enumerate(selected) if j!=i and s['gold']==r['gold'] and s['en']['link']!=r['en']['link']]
  assert others;r['unrelated_same_label']=others[0]
 (R/'inputs/intervention.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in selected))
 write(R/'protocol.json',dict(status='FROZEN_BEFORE_NEW_EXTRACTION_AND_INTERVENTION',models=prior['models'],identities=prior['identities'],base_identity=prior['base_identity'],languages=prior['languages'],layers=[6,12,18,24,30,36],rank=64,language_rank=128,fit=1024,dev=256,readouts=['complete_user_body_mean_L18','answer_prediction_boundary_six_layers'],primary_layer=18,primary_alpha=.5,alphas=[0,.5,1],random_seeds=[1115,1116,1117,1118,1119],equal_norm_scale_cap=10,intervention_n=128,intervention_selection='seed20261116 shuffle all900, first128 distinct EN passage texts, no result-based filtering; source linked by URL and question number; all labels aligned',intervention_scope='Previously exposed Belebele test subset, exploratory causal explanation only, not independent confirmation and never used to select layer/rank/dose. Extra EN input is not normal monolingual performance.',framework='Use frozen upstream task prompt, choices and process_results; custom activation hook and final-token likelihood, cross-check actual unmodified HFLM loglikelihood on same loaded model.',fit_scope='Original1024 training-side auxiliary IDs; dev is existing256 historical dev inputs, no fresh confirmation claim.',input_hashes={p.name:sha(p) for p in (R/'inputs').glob('*.jsonl')},source_hashes={str(p):sha(p) for p in sources.values()},old_analysis=str(OLD),limitations=['Single training seed','A has historical gradient checkpointing implementation difference from B/S/E/D','No new SFT; mechanism diagnostics do not establish new algorithm benchmark improvement','Existing geometry used512 fit rows and all its dev results already exposed; this is a follow-up','No claim to human-audited bilingual minimal-pair relations; authentic MCQA options provide task-content discrimination']))
 print('prepared',len(rows),len(selected),counts,flush=True)
if __name__=='__main__':main()
