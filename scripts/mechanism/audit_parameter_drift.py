"""Layer/component parameter drift; descriptive, not causal importance."""
import argparse,csv,json,math,platform,time
from pathlib import Path
import torch
from weights import load_pair,drift_rows,discover_components


def aggregate(rows,structure):
    output=[]
    for item in structure:
        for kind in ['block','attention','mlp',*item['projections']]:
            prefix=item.get(kind,item['projections'].get(kind));selected=[r for r in rows if r['parameter'].startswith(prefix+'.')]
            n=sum(r['elements'] for r in selected);changed=sum(r['changed_elements'] for r in selected)
            bn=sum(r['base_norm']**2 for r in selected)**.5;dn=sum(r['delta_norm']**2 for r in selected)**.5
            output.append(dict(layer=item['index'],component=kind,parameters=n,changed_elements=changed,changed_fraction=changed/n if n else None,base_norm=bn,delta_norm=dn,relative_norm=dn/(bn+1e-12)))
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--final',required=True);p.add_argument('--kind',choices=['full','lora'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(a.threads)
    base,final=load_pair(a.base,a.final,a.kind);structure=discover_components(final);rows=drift_rows(base,final);groups=aggregate(rows,structure)
    report={'scope':'Parameter change correlation only; no task or mechanism claim','base':str(Path(a.base).resolve()),'final':str(Path(a.final).resolve()),'kind':a.kind,'effective_weight_dtype':'float32','lora_rule':'PEFT safe_merge in float32, actual scaling; not A/B subtraction; BF16 deployed rounding may differ','unique_elements':sum(r['elements'] for r in rows),'architecture':structure,'parameters':rows,'groups':groups,'elapsed_seconds':time.monotonic()-start,'torch':torch.__version__,'python':platform.python_version(),'gpu_used':False}
    (a.output/'parameter_drift.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (a.output/'parameter_drift.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(groups[0]));w.writeheader();w.writerows(groups)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for comp in ['attention','mlp']:
        r=[v for v in groups if v['component']==comp];plt.plot([v['layer'] for v in r],[v['relative_norm'] for v in r],label=comp)
    plt.xlabel('Layer');plt.ylabel('Relative effective-weight change');plt.legend();plt.tight_layout();plt.savefig(a.output/'parameter_drift.png',dpi=160);plt.close()
    print(json.dumps({k:report[k] for k in ['unique_elements','elapsed_seconds','gpu_used']},ensure_ascii=False))
if __name__=='__main__':main()
