"""Describe saved official strict/flexible math results, without re-scoring.

Illustrations are the first three doc IDs satisfying an explicit diagnostic
category, never a model/metric selection. No inference or training is run.
"""
import argparse
import hashlib
import json
from pathlib import Path

LOG=Path('/path/to/workspace/logs/qwen3-multilingual/sft-v1')


def analyze(run,task,total):
    assert (run/'exit_code.txt').read_text().strip()=='0'
    files=list(run.glob('results/**/samples_'+task+'_*.jsonl'));assert len(files)==1
    source=files[0];groups={'strict-match':{},'flexible-extract':{}}
    for line in source.open():
        row=json.loads(line);records=groups[row['filter']];assert row['doc_id'] not in records
        records[row['doc_id']]=row
    strict,flex=groups['strict-match'],groups['flexible-extract']
    assert strict.keys()==flex.keys() and len(strict)==total
    expected_prefix='#### ' if task=='sft_v1_gsm8k' else 'Jibu ni '
    counts={k:0 for k in ['both_correct','strict_only_correct','flexible_only_correct','both_wrong',
        'strict_invalid','flexible_invalid','strict_invalid_flexible_correct','strict_invalid_without_expected_prefix','empty_response']}
    examples=[]
    for doc_id in sorted(strict):
        a,b=strict[doc_id],flex[doc_id]
        assert a['doc']==b['doc'] and a['arguments']==b['arguments'] and a['resps']==b['resps']
        s,f=int(a['exact_match']),int(b['exact_match']);assert s in (0,1) and f in (0,1)
        counts[{(1,1):'both_correct',(1,0):'strict_only_correct',(0,1):'flexible_only_correct',(0,0):'both_wrong'}[s,f]]+=1
        invalid=a['filtered_resps']==['[invalid]'];counts['strict_invalid']+=invalid
        counts['flexible_invalid']+=b['filtered_resps']==['[invalid]']
        counts['strict_invalid_flexible_correct']+=invalid and f==1
        text=a['resps'][0][0];counts['empty_response']+=not text.strip()
        counts['strict_invalid_without_expected_prefix']+=invalid and expected_prefix not in text
        if invalid and f and len(examples)<3:
            examples.append({'doc_id':doc_id,'target':a['target'],'official_strict_filtered':a['filtered_resps'],
                'official_flexible_filtered':b['filtered_resps'],'response_excerpt':text[:1500],
                'excerpt_truncated':len(text)>1500})
    strict_correct=counts['both_correct']+counts['strict_only_correct'];flex_correct=counts['both_correct']+counts['flexible_only_correct']
    results=list(run.glob('results/**/results_*.json'));assert len(results)==1
    official=json.loads(results[0].read_text())['results'][task]
    assert abs(official['exact_match,strict-match']-strict_correct/total)<1e-12
    assert abs(official['exact_match,flexible-extract']-flex_correct/total)<1e-12
    return {'expected_literal_prefix':expected_prefix,'samples':total,'strict_correct':strict_correct,'flexible_correct':flex_correct,
        'counts':counts,'first_three_strict_invalid_but_flexible_correct':examples,
        'samples_file':str(source),'samples_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,choices=[42,43,44],default=42);a=p.parse_args();assert not a.output.exists()
    methods={}
    for method in ['base','full','lora','ssu']:
        run=LOG/('base-core-other-core' if method=='base' else ('final-'+method+'-core-other' if a.seed==42 else f'seed{a.seed}-{method}-core-other-core'))
        methods[method]={task:analyze(run,task,total) for task,total in [('sft_v1_gsm8k',1319),('sft_v1_mgsm_native_cot_sw',250)]}
    report={'training_seed':a.seed,'status':'saved official scores unchanged; paired filter categories verified','inference_started':False,
        'interpretation':'Flexible match is auxiliary extraction, not a replacement primary score or proof of valid reasoning. Strict-invalid yet flexible-correct illustrates format contribution; flexible accuracy changes still remain task score changes.',
        'example_selection':'First3 sorted doc_ids per model/task with strict-invalid and flexible-correct; descriptive only; no tuning.',
        'methods':methods}
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for method,tasks in methods.items():
        for task,r in tasks.items():print(method,task,json.dumps({k:v for k,v in r.items() if k in ['samples','strict_correct','flexible_correct','counts']}))


if __name__=='__main__':main()
