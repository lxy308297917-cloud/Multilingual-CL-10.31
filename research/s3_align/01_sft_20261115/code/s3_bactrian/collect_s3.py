"""Audit actual S3 outputs, reuse upstream scores, and verify paired comparisons.

This is a post-processing addition.  It never edits the frozen running code,
launches a model, creates missing scores, or selects a checkpoint.
"""
import argparse
import contextlib
import csv
import importlib.util
import sys
import functools
from common import *
import numpy as np


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


shared=module('frozen_bactrian_collector',OLD/'collect_results.py')
sys.modules['collect_results']=shared
code_audit=module('frozen_code_collector',W/'work/benchmark-expansion-20261113/collect_results.py')
LEGACY_LOG=shared.LOG
LEGACY_CODE_LOG=code_audit.LOG
TASKS=shared.TASKS+[('en','HumanEval+',164)]
MODELS=['base','en','A','B','S','E','D']
CONTRASTS=[('base','en'),('base','A'),('en','A')]
CONTRASTS += [('base',x) for x in ['B','S','E','D']]
CONTRASTS += [('A',x) for x in ['B','S','E','D']]
CONTRASTS += [('B','E'),('S','E'),('D','E'),('B','D')]
GROUPS={
    'core_target':{('sw','Belebele'),('sw','Global-MMLU'),('en->sw','FLORES'),('sw->en','FLORES')},
    'english':{(l,t) for l,t,n in TASKS if l=='en'},
    'supplementary':{('sw','XL-Sum'),('ar','Belebele'),('ru','Belebele')},
    'reading':{(l,t) for l,t,n in TASKS if t=='Belebele'},
}


def legacy_name(model):return 'sw' if model=='A' else model


def stage_name(lang,task):
    if task=='FLORES':return 'flores_'+lang.replace('->','_')
    return {'Belebele':'reading','Global-MMLU':'gmmlu','MMLU':'mmlu',
            'XL-Sum':'summary','HumanEval+':'humaneval'}.get(task,'core')


@functools.lru_cache(maxsize=8)
def trained_identity(model):
    adapter=K/model/'sw_final_v1'
    done=read(adapter/'TRAINING_DONE.json');digest=sha(adapter/'adapter_model.safetensors')
    assert done['status']=='TRAINING_DONE' and done['weights_sha256']['adapter_model.safetensors']==digest
    assert done['steps']==4089 and done['samples']==65415 and done['main_sft_order_matches_historical_baseline']
    assert done['accounting']['global_counts']['shifted_assistant_tokens']==14314363
    assert done['execution_lock_sha256']==sha(J/'EXECUTION_LOCK.json')
    reload=read(adapter/'RELOAD_VERIFIED.json')
    assert reload['weights_sha256']==done['weights_sha256']
    assert reload['status']=='EXACT_ADAPTER_AND_REAL_INPUT_FORWARD_VERIFIED'
    if model=='S':assert done['old_total_input_tokens']==0
    else:assert done['old_total_input_tokens']>0
    if model!='B':
        assert done['alignment_total_input_tokens']>0
        smoke=read(K/model/'smoke_v1/SMOKE_DONE.json')
        assert smoke['alignment_gradient_probe']['nonzero_alignment_gradient_norm']>0
    return adapter,digest


def check_new_stage(model,folder,code=False):
    adapter,digest=trained_identity(model)
    done=read(folder/'DONE.json');plan=read(folder/'PLAN.json')
    assert plan['adapter_sha256']==digest
    if not code:
        assert done['adapter_sha256']==digest and done['endpoint']=='sw'
        assert done['status']=='TASK_EXECUTION_COMPLETE_PENDING_SCORE_AUDIT'
    else:assert done['status']=='ACTUAL_CODE_EXECUTION_COMPLETE_PENDING_PAIR_AUDIT'
    return adapter,digest


def generation_context(folder):
    m=read(folder/'manifest.json')
    # Parameters affecting generation and data identity must match, including
    # actual prompt-token hashes. Adapter paths naturally differ across models.
    keys=['benchmark_protocol','backend','batch_size','source_files_sha256',
          'data_manifest_hash','prompt_tokens_sha256']
    assert all(k in m for k in keys)
    context={k:m[k] for k in keys}
    proof=read(O/'GENERATION_CODE_EQUIVALENCE.json')
    assert proof['status']=='IDENTICAL_GENERATION_IMPLEMENTATION_EXCEPT_PREALLOCATION_GPU_IDENTITY_ASSERTION'
    assert all(sha(p)==digest for p,digest in proof['sources'].items())
    code=dict(context['source_files_sha256'])
    if code['generate.py'] in proof['sources'].values():
        # Both exact source identities remain in the input manifests/proof.
        # Canonicalize only this audited GPU-identity assertion difference.
        code['generate.py']='generation_ast:'+proof['shared_ast_sha256']
    context['source_files_sha256']=code
    return context


def find(model,lang,task):
    old=model in ['base','en','A']
    if task=='HumanEval+':
        if old:return code_audit.code(legacy_name(model),'humaneval','plus')
        folder=LOG/'evaluation'/model/'sw/humaneval'
        if not (folder/'execution/DONE.json').exists() or not (folder/'DONE.json').exists():return None
        previous_log,previous_check=code_audit.LOG,code_audit.check_stage_identity
        try:
            code_audit.LOG=LOG/'evaluation'/model
            code_audit.check_stage_identity=lambda endpoint,root:check_new_stage(model,root,code=True)
            return code_audit.code('sw','humaneval','plus')
        finally:code_audit.LOG,code_audit.check_stage_identity=previous_log,previous_check
    if old:
        found=shared.locate(legacy_name(model),lang,task)
    else:
        folder=LOG/'evaluation'/model/'sw'/stage_name(lang,task)
        if not (folder/'DONE.json').exists():return None
        adapter,digest=check_new_stage(model,folder)
        previous=shared.LOG
        try:
            shared.LOG=LOG/'evaluation'/model
            found=shared.locate('sw',lang,task)
        finally:shared.LOG=previous
    if found is None:return None
    vector,sources=found
    context=shared.protocol_context(sources)
    if task in ['FLORES','XL-Sum']:
        if task=='FLORES':
            generation=Path(sources[0]['path']).parent.parent/'generation'
        elif model=='base':
            generation=shared.OLD/'base-generation-v2/xlsum_sw_v2'
        else:generation=Path(sources[0]['path']).parent.parent/'generation'
        context=generation_context(generation)
        sources=sources+[dict(path=str(generation/'manifest.json'),sha256=sha(generation/'manifest.json'),
                              samples=len(vector['rows']) if task=='FLORES' else len(vector))]
    if not old:
        if task in ['FLORES','XL-Sum']:
            identity=read(generation/'manifest.json')['checkpoint_identity']
            assert identity[str((adapter/'adapter_model.safetensors').resolve())]==digest
        else:
            r=shared.result_json(str(Path(sources[0]['path']).parent))
            ma=r['config']['model_args']
            assert Path(ma['pretrained']).resolve()==BASE.resolve()
            assert ma.get('peft',ma.get('lora_local_path'))==str(adapter)
    return vector,sources,context


def write_csv(path,rows,fields):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else fields)
        writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    ap.add_argument('--group',choices=['all',*GROUPS],default='all')
    ap.add_argument('--partial',action='store_true')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    tasks=[x for x in TASKS if a.group=='all' or x[:2] in GROUPS[a.group]]
    vectors={};contexts={};scores=[];provenance=[];missing=[];intervals=[]
    baseline=read(W/'outputs/bactrian_single_20261112/audit-final-main-20261113/AUDIT.json')
    assert baseline['status']=='COMPLETE_RAW_SCORE_AND_PAIR_AUDIT' and baseline['base_identity_gate_ready']
    for model in a.models:
        for lang,task,n in tasks:
            found=find(model,lang,task)
            if found is None:missing.append([model,lang,task]);continue
            v,files,context=found;key=model,lang,task
            if task=='FLORES':score=v['score']
            else:
                assert len(v)==n
                score=float(np.mean([r['value'] for r in v.values()]))
            vectors[key]=v;contexts[key]=context
            scores.append(dict(model=model,language=lang,task=task,samples=n,score=score,
                               status='HISTORICAL_REAL_OUTPUT_VERIFIED' if model in ['base','en','A'] else 'NEW_REAL_OUTPUT_VERIFIED'))
            provenance.append(dict(model=model,language=lang,task=task,files=files,protocol=context))
    for ci,(left,right) in enumerate(CONTRASTS):
        if left not in a.models or right not in a.models:continue
        for ti,(lang,task,n) in enumerate(TASKS):
            ka,kb=(left,lang,task),(right,lang,task)
            if ka not in vectors or kb not in vectors:continue
            assert contexts[ka]==contexts[kb],('Protocol mismatch',ka,kb,contexts[ka],contexts[kb])
            seed=np.random.SeedSequence([42,ci,ti])
            compare=shared.mt_paired if task=='FLORES' else shared.audit.paired
            result=compare(vectors[ka],vectors[kb],seed)
            intervals.append(dict(before=left,after=right,language=lang,task=task,**result))
    write_csv(a.output/'scores.csv',scores,['model','language','task','samples','score','status'])
    write_csv(a.output/'paired_changes.csv',intervals,['before','after','language','task','delta','ci95_low','ci95_high'])
    dump(a.output/'provenance.json',provenance)
    numeric=a.output/'numeric';numeric.mkdir()
    for (model,lang,task),v in vectors.items():
        if task=='FLORES':continue
        keys=sorted(v,key=str)
        dump(numeric/f'{model}__{lang}__{task}.json',dict(model=model,language=lang,task=task,
            item_ids=[str(k) for k in keys],identity=[v[k]['identity'] for k in keys],
            clusters=[v[k]['cluster'] for k in keys],values=[v[k]['value'] for k in keys]))
    expected_intervals=sum(l in a.models and r in a.models for l,r in CONTRASTS)*len(tasks)
    report=dict(status='PARTIAL' if missing else 'REQUESTED_SCOPE_RAW_AND_PAIRED_AUDIT_COMPLETE',
        requested_models=a.models,requested_group=a.group,verified_cells=len(scores),
        required_cells=len(a.models)*len(tasks),paired_intervals=len(intervals),required_intervals=expected_intervals,
        missing=missing,full_goal_scope=(set(a.models)==set(MODELS) and a.group=='all'),
        limitations=['One training seed; paired intervals measure fixed-test uncertainty, not training-seed robustness.',
                     'No multiplicity correction. No test-based hyperparameter or checkpoint selection.',
                     'Historical GSM8K extraction and IFEval checker limitations retained.',
                     'AR/RU reading and SW XL-Sum are supplementary, not the shared target-language core.'],
        analysis_source_sha256=sha(__file__),training_execution_lock_sha256=sha(J/'EXECUTION_LOCK.json'))
    dump(a.output/'AUDIT.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if not a.partial:
        assert not missing and len(intervals)==expected_intervals


if __name__=='__main__':main()
