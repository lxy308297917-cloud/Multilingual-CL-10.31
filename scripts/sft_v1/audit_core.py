"""Check complete saved core results and aggregate the official per-sample scores.

This is an output completeness check, not a new benchmark scorer. Official
Harness/BenchMAX scores stay unchanged; no score threshold selects a model.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

LOG=Path('/path/to/workspace/logs/qwen3-multilingual/sft-v1')


def one(paths):
    paths=list(paths)
    assert len(paths)==1, [str(p) for p in paths]
    return paths[0]


def audit(runs=None):
    runs = runs or {tag: f"base-core-{tag}" for tag in ("prox-en", "prox-sw", "other-core")}
    expected=json.loads((LOG/'evaluation_environment_preflight.json').read_text())['tasks']
    report={}
    for tag in ['prox-en','prox-sw','other-core']:
        run=LOG/runs[tag]
        assert (run/'exit_code.txt').read_text().strip()=='0',str(run)
        result_path=one(run.glob('results/**/results_*.json'))
        result=json.loads(result_path.read_text())
        selected=[t for t in expected if (t['task'].startswith('sft_v1_mmlu_prox_en_') if tag=='prox-en' else t['task'].startswith('sft_v1_mmlu_prox_sw_') if tag=='prox-sw' else not t['task'].startswith('sft_v1_mmlu_prox_'))]
        for task in selected:
            name,n=task['task'],task['rows'];summary=result['results'][name]
            assert summary['sample_len']==n,(name,summary['sample_len'],n)
            samples=one(result_path.parent.glob(f'samples_{name}_*.jsonl'))
            filters=defaultdict(list)
            for line in samples.open():
                row=json.loads(line);filters[row['filter']].append(row)
            metrics={}
            diagnostics={}
            for label,rows in filters.items():
                assert len(rows)==n and len({r['doc_id'] for r in rows})==n,(name,label)
                for metric in rows[0]['metrics']:
                    values=[v for r in rows for v in (r[metric] if isinstance(r[metric],list) else [r[metric]])]
                    assert values and all(v in (0,1) for v in values),(name,metric)
                    rate=sum(values)/len(values)
                    assert abs(rate-summary[f'{metric},{label}'])<1e-10,(name,metric,label)
                    metrics[f'{metric},{label}']={'correct':int(sum(values)),'denominator':len(values),'rate':rate}
                diagnostics[label]={'answer_extraction_failures':sum(r['filtered_resps'] in (['[invalid]'],[''],[None]) for r in rows),'mean_response_characters':sum(len(r['resps'][0][0]) for r in rows)/n,'empty_outputs':sum(not r['resps'][0][0].strip() for r in rows)}
            report[name]={'samples':n,'fewshot':task['shots'],'metrics':metrics,'format_diagnostics':diagnostics,'result_file':str(result_path),'samples_file':str(samples)}
        meta=[json.loads(line) for line in (run/'completion-metadata.jsonl').open()]
        assert len(meta)==sum(t['rows'] for t in selected),(tag,len(meta))
        assert len({m['request_id'] for m in meta})==len(meta)
        report[f'backend_{tag}']={'requests':len(meta),'normal_stop':sum(m['finish_reason']=='stop' for m in meta),'length_limit':sum(m['finish_reason']=='length' for m in meta),'mean_generated_tokens':sum(m['generated_tokens'] for m in meta)/len(meta),'other_finish_reasons':sorted({str(m['finish_reason']) for m in meta if m['finish_reason'] not in ('stop','length')})}
        if tag in ['prox-en','prox-sw']:
            group='sft_v1_mmlu_prox_'+tag[-2:]
            key='exact_match,custom-extract'
            count=sum(report[t['task']]['metrics'][key]['correct'] for t in selected)
            total=sum(t['rows'] for t in selected)
            assert total==11759
            group_summary=result.get('groups',{}).get(group,result['results'].get(group))
            assert group_summary and abs(count/total-group_summary[key])<1e-10
            report[group]={'correct':count,'samples':total,'exact_match':count/total}
    assert len([k for k in report if k.startswith('sft_v1_') and k not in ('sft_v1_mmlu_prox_en','sft_v1_mmlu_prox_sw')])==32
    return {'status':'validated complete saved outputs and official score aggregation','no_score_threshold_or_model_selection':True,'tasks':report}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prox-en-run',default='base-core-prox-en')
    p.add_argument('--prox-sw-run',default='base-core-prox-sw')
    p.add_argument('--other-run',default='base-core-other-core')
    a=p.parse_args()
    runs={'prox-en':a.prox_en_run,'prox-sw':a.prox_sw_run,'other-core':a.other_run}
    assert not a.output.exists()
    a.output.write_text(json.dumps(audit(runs),indent=2,ensure_ascii=False)+'\n')
