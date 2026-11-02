"""Same-input Base/Final activation drift with explicit prompt/reference regions."""
import argparse,collections,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from weights import load_pair,discover_components
from probes import load_probes
from inference import encode_probe
from activations import capture,compare_same_input,linear_cka


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--final',required=True);p.add_argument('--kind',choices=['full','lora'],required=True);p.add_argument('--probes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-per-group',type=int,default=4);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--gpu-uuid');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.device=='cuda':
        if not a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid:raise ValueError('Explicit single UUID required')
        check=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'preflight_gpu.py'),'--uuid',a.gpu_uuid,'--max-user-gpus','4'],capture_output=True,text=True);(a.output/'gpu_preflight.json').write_text(check.stdout);check.check_returncode()
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42);rows=load_probes(a.probes);groups=collections.defaultdict(list)
    for r in rows:groups[(r['task'],r['input_language'],r['output_language'])].append(r)
    selected=[r for key,group in sorted(groups.items()) for r in sorted(group,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:a.max_per_group]]
    tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True);base,final=load_pair(a.base,a.final,a.kind,device=a.device);base.to(a.device);structure=discover_components(final);names=[item[k] for item in structure for k in ['block','attention','mlp']];module_metadata={item[k]:(item['index'],k) for item in structure for k in ['block','attention','mlp']}
    metrics=[];pooled=collections.defaultdict(lambda:[[],[]])
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
    for row in selected:
        prefix,identity=encode_probe(tok,row);answer=row['answer'];text=(answer[0] if isinstance(answer,list) else str(answer)) if answer is not None else '';target=tok.encode(text,add_special_tokens=False) if text else [];ids=torch.tensor([prefix+target],device=a.device)
        with torch.inference_mode():
            with capture(base,names) as source:base(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
            with capture(final,names) as destination:final(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
        for name in names:
            for region,left,right in [('prompt',0,len(prefix)),('teacher_forced_reference',len(prefix),len(prefix)+len(target))]:
                if left==right:continue
                x=source[name][:,left:right,:];y=destination[name][:,left:right,:];layer,component=module_metadata[name]
                result={'id':row['id'],'pair_id':row['pair_id'],'task':row['task'],'input_language':row['input_language'],'output_language':row['output_language'],'layer':layer,'component':component,'region':region,'prompt_token_sha256':identity['prompt_token_sha256'],**compare_same_input(x,y)};metrics.append(result)
                key=(row['task'],row['input_language'],row['output_language'],layer,component,region);pooled[key][0].append(x.mean(1)[0]);pooled[key][1].append(y.mean(1)[0])
        del source,destination
    aggregate=[]
    for key,(xs,ys) in sorted(pooled.items()):
        selected_metrics=[r for r in metrics if (r['task'],r['input_language'],r['output_language'],r['layer'],r['component'],r['region'])==key]
        aggregate.append(dict(task=key[0],input_language=key[1],output_language=key[2],layer=key[3],component=key[4],region=key[5],samples=len(xs),mean_cosine=sum(r['mean_token_cosine'] for r in selected_metrics)/len(xs),mean_relative_l2=sum(r['relative_l2'] for r in selected_metrics)/len(xs),linear_cka=linear_cka(torch.stack(xs),torch.stack(ys)) if len(xs)>1 else None))
    report={'scope':'Same-input Base/Final descriptive drift. Teacher-forced reference is NOT generated output. No cross-language token alignment or causal importance claim.','probes_sha256':hashlib.sha256(a.probes.read_bytes()).hexdigest(),'selection':'First max-per-group by SHA256(id), no model-score selection','selected_ids':[r['id'] for r in selected],'sample_count':len(selected),'analysis_dtype':'float32','regions':['prompt','teacher_forced_reference when answer available'],'cka_limit':'Very small sample groups can yield unstable or degenerate CKA; descriptive only. Do not infer equivalence.','elapsed_seconds':time.monotonic()-start,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else 0,'rows':aggregate}
    (a.output/'activation_drift.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(a.output/'activation_drift_samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in metrics))
    with (a.output/'activation_drift.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,7))
    keys=sorted({(r['task'],r['input_language'],r['output_language']) for r in aggregate})
    for key in keys:
        values=[r for r in aggregate if (r['task'],r['input_language'],r['output_language'])==key and r['component']=='block' and r['region']=='prompt'];ax.plot([r['layer'] for r in values],[r['mean_relative_l2'] for r in values],label='/'.join(key))
    ax.set_xlabel('Layer');ax.set_ylabel('Same-input prompt activation relative L2');ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(a.output/'activation_drift_by_task.png',dpi=160);plt.close(fig)
    print(json.dumps({'sample_count':len(selected),'rows':len(aggregate),'elapsed_seconds':report['elapsed_seconds']}))
if __name__=='__main__':main()
