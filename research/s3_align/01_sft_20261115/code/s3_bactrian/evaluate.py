"""Reuse the successful Bactrian execution protocol with isolated outputs."""
import argparse
import csv
import datetime
import fcntl
import importlib.util
import os
import subprocess
import sys
from common import *


def stage_scores(root,stage):
    values=[]
    if stage in ['flores_en_sw','flores_sw_en','summary']:
        summary=read(root/'scores/summary.json')
        metric='ROUGE-Lsum' if stage=='summary' else 'chrF++'
        language='sw' if stage=='summary' else ('en->sw' if stage=='flores_en_sw' else 'sw->en')
        values.append(dict(language=language,task='XL-Sum' if stage=='summary' else 'FLORES',
                           metric=metric,score=summary['metrics'][metric],samples=summary['samples']))
    elif stage=='humaneval':
        s=read(root/'execution/scores.json')
        values.append(dict(language='en',task='HumanEval+',metric='pass@1',score=s['plus_pass_at_1'],samples=s['samples']))
    elif stage in ['mmlu','gmmlu']:
        s=read(root/'raw-accuracy-audit.json')
        values.append(dict(language='en' if stage=='mmlu' else 'sw',task='MMLU' if stage=='mmlu' else 'Global-MMLU',
                           metric='ACC',score=100*s['micro_accuracy'],samples=14042))
    else:
        files=list(root.glob('results/**/results_*.json'));assert len(files)==1
        s=read(files[0])
        mappings=({'belebele_swh_Latn_locked':('sw','Belebele','acc,none',900),
                   'belebele_eng_Latn_locked':('en','Belebele','acc,none',900)} if stage=='reading' else
                  {'sft_v1_gsm8k':('en','GSM8K','exact_match,flexible-extract',1319),
                   'sft_v1_ifeval':('en','IFEval-prompt','prompt_level_strict_acc,none',541)})
        for task,(lang,label,key,n) in mappings.items():
            values.append(dict(language=lang,task=label,metric=key,score=100*s['results'][task][key],samples=n))
    with (W/'outputs/bactrian_single_20261112/audit-final-main-20261113/scores.csv').open() as f:
        old=list(csv.DictReader(f))
    for row in values:
        for model,label in [('base','base'),('sw','ordinary_lora')]:
            matched=[r for r in old if (r['model'],r['language'],r['task'])==(model,row['language'],row['task'])]
            if matched:
                row[label+'_score']=float(matched[0]['score'])
                row['delta_vs_'+label]=row['score']-row[label+'_score']
        if row['task']=='HumanEval+':
            source=W/'outputs/benchmark_expansion_20261113/audit-code-complete-20261113/scores.csv'
            with source.open() as f: code_rows=list(csv.DictReader(f))
            for model,label in [('base','base'),('sw','ordinary_lora')]:
                old_code=[r for r in code_rows if (r['model'],r['task'],r['metric'])==(model,'humaneval','plus')]
                assert len(old_code)==1 and old_code[0]['status']=='REAL_RAW_UPSTREAM_SCORE_VERIFIED'
                row[label+'_score']=float(old_code[0]['score'])
                row['delta_vs_'+label]=row['score']-row[label+'_score']
            row['baseline_source']=str(source)
    return values


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['B','S','D','E'],required=True);a=ap.parse_args()
    assert (K/a.arm/'sw_final_v1/RELOAD_VERIFIED.json').exists()
    spec=importlib.util.spec_from_file_location('frozen_bactrian_eval',OLD/'evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.K=K/a.arm
    module.LOG=LOG/'evaluation'/a.arm
    # Finish target tasks first and surface every completed stage before MMLU.
    stages=['reading','flores_en_sw','flores_sw_en','gmmlu','summary','core','humaneval','mmlu']
    for stage in stages:
        if stage=='humaneval':
            ext=W/'work/benchmark-expansion-20261113'
            root=module.LOG/'sw'/stage
            root.mkdir(parents=True,exist_ok=False)
            adapter=K/a.arm/'sw_final_v1'
            env=dict(os.environ,VLLM_USE_V1='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
                VLLM_WORKER_MULTIPROC_METHOD='spawn')
            command=[module.E,str(module.ROOT/'scripts/representation_benchmarks_20261109/generate_code.py'),
                '--dataset',str(ext/'code-assets-v1/humaneval_plus_v2'),'--model',str(BASE),
                '--tokenizer',str(BASE),'--method','lora','--adapter',str(adapter),
                '--gpu-uuid',os.environ['CUDA_VISIBLE_DEVICES'],'--output',str(root/'generation'),
                '--scope','full','--batch-size','8','--max-user-gpus','4']
            scoring=[PY,str(ext/'execute_code.py'),'--generation',str(root/'generation'),
                     '--output',str(root/'execution'),'--dataset','humaneval']
            dump(root/'PLAN.json',dict(command=command,scoring=scoring,
                adapter_sha256=sha(adapter/'adapter_model.safetensors'),
                protocol='Historical full164 HumanEvalPlus; actual existing isolated EvalPlus executor.'))
            with (root/'generation.log').open('x') as f:
                subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            with (O/'code-execution.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                with (root/'scoring.log').open('x') as f:
                    subprocess.run(scoring,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
            dump(root/'DONE.json',dict(status='ACTUAL_CODE_EXECUTION_COMPLETE_PENDING_PAIR_AUDIT'))
        else:
            sys.argv=[str(__file__),'--endpoint','sw','--stage',stage]
            module.main()
        event=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                   arm=a.arm,stage=stage,status='EXECUTION_COMPLETE_PENDING_RAW_COMPARISON',
                   path=str(module.LOG/'sw'/stage),scores=stage_scores(module.LOG/'sw'/stage,stage))
        with (O/'MILESTONES.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
        print(json.dumps(event,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
