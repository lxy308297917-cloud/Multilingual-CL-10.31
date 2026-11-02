"""Post-hoc numeric response audit. Original frozen scores remain unchanged.

Only explicit answer markers or short, completed, single-number responses are
resolved. Other cases are unknown, with cohort-level accuracy bounds rather
than a rescued benchmark score. Not a semantic proof of complete reasoning.
"""
import argparse,csv,json,re
from decimal import Decimal,InvalidOperation
from pathlib import Path
NUMBER=r'[-+]?\d[\d,]*(?:\.\d+)?'


def resolve_numeric_response(text,gold,completed,word_limit=20):
    strict=re.findall(r'####\s*('+NUMBER+r')',text)
    tagged=re.findall(r'<answer>\s*('+NUMBER+r')\s*</answer>',text)
    numbers=re.findall(NUMBER,text)
    if strict:value=strict[-1];rule='explicit_last_answer_marker'
    elif tagged:value=tagged[-1];rule='explicit_answer_tag'
    elif completed and len(numbers)==1 and len(text.split())<=word_limit:
        value=numbers[0];rule='completed_short_single_number'
    else:return {'resolved':False,'correct':None,'extracted':None,'rule':'unresolved_without_unambiguous_scalar','lower_correct':0,'upper_correct':1}
    try:correct=Decimal(value.replace(',',''))==Decimal(str(gold).replace(',',''))
    except InvalidOperation:raise ValueError('Numeric annotation not decimal')
    return {'resolved':True,'correct':bool(correct),'extracted':value,'rule':rule,'lower_correct':int(correct),'upper_correct':int(correct)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--probes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--eos-id',type=int,required=True);a=p.parse_args();done=json.loads((a.run/'DONE.json').read_text());a.output.mkdir(parents=True,exist_ok=False)
    with a.probes.open() as stream:probes={r['id']:r for r in map(json.loads,stream)}
    records=[];summary=[]
    for condition in done['conditions']:
        with (a.run/condition/'predictions.jsonl').open() as stream:rows=[r for r in map(json.loads,stream) if r['task']=='math']
        result=[]
        for r in rows:
            completed=bool(r['generated_token_ids']) and r['generated_token_ids'][-1]==a.eos_id;diagnostic=resolve_numeric_response(r['raw_response'],probes[r['id']]['answer'],completed)
            item={'condition':condition,'id':r['id'],'gold':probes[r['id']]['answer'],'raw_response':r['raw_response'],'original_content_score':r['score']['content_correct'],'original_format_score':r['score']['format_passed'],'hit_generation_cap':r['generation_hit_cap'],'completed_with_eos':completed,**diagnostic};result.append(item);records.append(item)
        n=len(result);summary.append({'condition':condition,'samples':n,'original_restricted_extraction_correct':sum(bool(r['original_content_score']) for r in result),'original_format_passed':sum(bool(r['original_format_score']) for r in result),'resolved_numeric_correct':sum(r['correct'] is True for r in result),'resolved_numeric_incorrect':sum(r['correct'] is False for r in result),'unresolved':sum(not r['resolved'] for r in result),'cohort_accuracy_lower_bound':sum(r['lower_correct'] for r in result)/n,'cohort_accuracy_upper_bound':sum(r['upper_correct'] for r in result)/n})
    (a.output/'math_response_audit.json').write_text(json.dumps({'scope':'Post-hoc diagnostic of frozen development generations, not changed primary scores or official benchmark; scalar extraction does not validate reasoning','rules':'Last explicit #### answer marker or answer tag; else EOS-completed response <=20 words with exactly one numeric expression; unresolved counted as unknown, not wrong. Commas normalized.','primary_scores_changed':False,'rows':summary},ensure_ascii=False,indent=2)+'\n');(a.output/'math_response_audit_samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    with (a.output/'math_response_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps({'conditions':len(summary),'samples_audited':len(records)}))
if __name__=='__main__':main()
