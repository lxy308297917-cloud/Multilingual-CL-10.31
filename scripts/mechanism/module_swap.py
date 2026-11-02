"""Predeclared few-module intervention runner, independent development only."""
import argparse,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from weights import load_pair,select_parameters,unique_entries,intervene,discover_components,validate_pair
from probes import load_probes
from inference import evaluate_rows,summarize


def selection_norm(base,final,names):
    b=dict(base.named_parameters());f=dict(final.named_parameters())
    return sum(float((f[n].detach().cpu().double()-b[n].detach().cpu().double()).square().sum()) for n,_,_ in unique_entries(final) if n in names)**.5


def resolve_plan(base,final,plan):
    resolved={};n_layers=len(discover_components(final))
    for item in plan['conditions']:
        label=item['id']
        if label in resolved or label in ('base','final'):raise ValueError('Duplicate/reserved condition id')
        if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):raise ValueError('Unsafe condition id')
        selector=item['selector']
        if 'random_like' in selector:
            reference=resolved[selector['random_like']];component=reference['component'];candidates=[]
            if len(reference['layers'])!=1:raise ValueError('Random control currently supports one-layer reference')
            for layer in range(n_layers):
                if layer in reference['layers']:continue
                names=select_parameters(final,[layer],component)
                count=sum(p.numel() for n,p,_ in unique_entries(final) if n in names)
                if count==reference['parameter_count']:candidates.append(layer)
            if not candidates:raise ValueError('No parameter-matched random module')
            layers=[random.Random(selector['seed']).choice(candidates)]
        else:layers=selector['layers'];component=selector['component']
        names=select_parameters(final,layers,component);count=sum(p.numel() for n,p,_ in unique_entries(final) if n in names)
        if item['mode'] not in ('restore','inject') or not 0<=item['alpha']<=1:raise ValueError('Invalid mode or alpha')
        matched_norm=resolved[item['norm_like']]['update_norm'] if item.get('norm_like') else None
        norm=selection_norm(base,final,names)
        if matched_norm is not None and norm==0 and matched_norm>0:raise ValueError('Cannot norm-match zero update')
        resolved[label]={'layers':layers,'component':component,'names':names,'parameter_count':count,'update_norm':norm,'matched_norm':matched_norm,'mode':item['mode'],'alpha':item['alpha'],'control_description':item.get('control_description')}
    return resolved


def save_condition(out,label,results,metadata):
    dest=out/label;dest.mkdir(exist_ok=False)
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    (dest/'summary.json').write_text(json.dumps({'condition':label,'groups':summarize(results),'metadata':metadata},ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--final',required=True);p.add_argument('--kind',choices=['full','lora'],required=True);p.add_argument('--probes',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--gpu-uuid');a=p.parse_args();rows=load_probes(a.probes);plan=json.loads(a.plan.read_text())
    if plan['formal_test_used_for_selection'] is not False or len(plan['conditions'])>24:raise ValueError('Require bounded, independent-development plan')
    a.output.mkdir(parents=True,exist_ok=False)
    if a.device=='cuda':
        if not a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid:raise ValueError('Explicit single UUID required')
        check=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'preflight_gpu.py'),'--uuid',a.gpu_uuid,'--max-user-gpus','4'],capture_output=True,text=True);(a.output/'gpu_preflight.json').write_text(check.stdout);check.check_returncode()
    torch.set_num_threads(2);torch.manual_seed(42);start=time.monotonic();tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True);base,final=load_pair(a.base,a.final,a.kind,device=a.device);validate_pair(base,final);resolved=resolve_plan(base,final,plan)
    max_new=plan['max_new_tokens'];max_input=plan['max_input_tokens'];base.to(a.device)
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
    protocol={'probes_sha256':hashlib.sha256(a.probes.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),'chat_template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),'enable_thinking':False,'do_sample':False,'use_model_defaults':False,'analysis_dtype':'float32','attention':'eager','max_new_tokens':max_new,'max_input_tokens':max_input,'max_new_tokens_by_task':plan.get('max_new_tokens_by_task',{}),'samples':len(rows),'source_note':'Diagnostics only, not official benchmark scores'}
    (a.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    done=[]
    def evaluate(model):return evaluate_rows(model,tokenizer,rows,max_new,max_input,plan.get('max_new_tokens_by_task'))
    for label,model in [('base',base),('final',final)]:
        results=evaluate(model)
        if label=='final':final_reference=results
        save_condition(a.output,label,results,{'role':'reference'});done.append(label);(a.output/'progress.json').write_text(json.dumps({'completed':done,'phase':'references'},indent=2)+'\n')
    base.to('cpu')
    for label,condition in resolved.items():
        c=condition;before=time.monotonic()
        with intervene(base,final,c['names'],c['mode'],c['alpha'],c['matched_norm']) as intervention:results=evaluate(final)
        assert intervention['restoration_verified']
        if c['mode']=='restore' and c['alpha']==1:assert results==final_reference,'Identity condition changed predictions'
        metadata={k:v for k,v in c.items() if k!='names'};metadata.update(selected_parameter_names=sorted(c['names']),intervention=intervention,elapsed_seconds=time.monotonic()-before)
        save_condition(a.output,label,results,metadata);done.append(label);(a.output/'progress.json').write_text(json.dumps({'completed':done,'phase':'interventions'},indent=2)+'\n');print(json.dumps({'condition':label,'completed':len(done),'elapsed_seconds':time.monotonic()-start}),flush=True)
    summary={'status':'completed','conditions':done,'samples_per_condition':len(rows),'elapsed_seconds':time.monotonic()-start,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else 0,'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if a.device=='cuda' else 0,'independent_functional_separation_established':False,'interpretation':'Conditional intervention effects require separate matrix and uncertainty analysis'}
    (a.output/'DONE.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
