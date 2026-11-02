"""Paired conditional effects and cluster bootstrap, separate from benchmarks."""
import argparse,csv,json
from pathlib import Path
import numpy as np


def cluster_interval(differences,clusters,repeats=2000,seed=42):
    if not differences or len(differences)!=len(clusters):raise ValueError('Unpaired observations')
    grouped={}
    for value,key in zip(differences,clusters):grouped.setdefault(key,[]).append(float(value))
    keys=sorted(grouped);point=float(np.mean(differences))
    if len(keys)<2:return point,None,None,len(keys)
    means=[];rng=np.random.default_rng(seed)
    for indices in rng.integers(0,len(keys),size=(repeats,len(keys))):
        values=[v for index in indices for v in grouped[keys[index]]];means.append(float(np.mean(values)))
    low,high=np.quantile(means,[.025,.975]);return point,float(low),float(high),len(keys)


def impact_rows(reference,condition,repeats=2000):
    ref={x['id']:x for x in reference};alt={x['id']:x for x in condition}
    if ref.keys()!=alt.keys() or len(ref)!=len(reference) or len(alt)!=len(condition):raise ValueError('Missing/duplicate sample IDs')
    groups={}
    for id,a in ref.items():
        b=alt[id]
        for key in ['task','input_language','output_language','prompt_token_sha256','max_new_tokens']:
            if a[key]!=b[key]:raise ValueError(f'Unmatched sample protocol: {id} {key}')
        for metric in ['content_correct','format_passed','language_correct']:
            av=a['score'][metric];bv=b['score'][metric]
            if (av is None)!=(bv is None):raise ValueError('Metric denominator changed')
            if av is None:continue
            key=(a['task'],a['input_language'],a['output_language'],metric)
            groups.setdefault(key,[]).append((float(av),float(bv),a.get('pair_id') or id))
    rows=[]
    for key,values in sorted(groups.items()):
        differences=[a-b for a,b,_ in values];effect,low,high,nclusters=cluster_interval(differences,[x[2] for x in values],repeats)
        rows.append(dict(task=key[0],input_language=key[1],output_language=key[2],metric=key[3],samples=len(values),clusters=nclusters,final_rate=100*np.mean([a for a,_,_ in values]),intervened_rate=100*np.mean([b for _,b,_ in values]),effect_final_minus_intervened_pp=100*effect,ci95_low_pp=100*low if low is not None else None,ci95_high_pp=100*high if high is not None else None))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args();done=json.loads((a.run/'DONE.json').read_text());a.output.mkdir(parents=True,exist_ok=False)
    def read(label):
        with (a.run/label/'predictions.jsonl').open() as stream:return [json.loads(x) for x in stream]
    final=read('final');all_rows=[]
    for condition in done['conditions']:
        if condition in ('base','final'):continue
        summary=json.loads((a.run/condition/'summary.json').read_text());metadata=summary['metadata'];intervention=metadata['intervention']
        if not intervention['restoration_verified']:raise ValueError('Intervention did not roll back')
        for r in impact_rows(final,read(condition),a.bootstrap):r.update(condition=condition,mode=metadata['mode'],alpha=metadata['alpha'],component=metadata['component'],layers=metadata['layers'],selected_parameters=metadata['parameter_count'],actual_changed_parameters=intervention['changed_unique_elements'],actual_replacement_norm=intervention['actual_replacement_norm']);all_rows.append(r)
    (a.output/'impact_matrix.json').write_text(json.dumps({'scope':'Conditional effect on frozen diagnostic probes, not benchmark or independent causal attribution','sign':'I=Final minus intervened. Positive: removing update harms this score. Negative: intervention improves score. Inject/half/random use same descriptive contrast, not all literal restore.','bootstrap':'Paired resampling by pair_id or sample ID; 95% percentile intervals; exploratory, no multiple-comparison correction; zero empirical interval is not proof of population equality','source_protocol':json.loads((a.run/'protocol.json').read_text()),'rows':all_rows},ensure_ascii=False,indent=2)+'\n')
    with (a.output/'impact_matrix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=sorted({(r['task'],r['input_language'],r['output_language'],r['metric']) for r in all_rows});conditions=[c for c in done['conditions'] if c not in ('base','final')];array=np.full((len(conditions),len(keys)),np.nan)
    for r in all_rows:array[conditions.index(r['condition']),keys.index((r['task'],r['input_language'],r['output_language'],r['metric']))]=r['effect_final_minus_intervened_pp']
    fig,ax=plt.subplots(figsize=(max(12,len(keys)*.45),max(5,len(conditions)*.4)));bound=max(1,float(np.nanmax(np.abs(array))));im=ax.imshow(array,cmap='coolwarm',vmin=-bound,vmax=bound,aspect='auto');ax.set_yticks(range(len(conditions)),conditions,fontsize=8);ax.set_xticks(range(len(keys)),['/'.join(k) for k in keys],rotation=90,fontsize=7);fig.colorbar(im,ax=ax,label='Final minus intervention (pp)');fig.tight_layout();fig.savefig(a.output/'impact_heatmap.png',dpi=160);plt.close(fig)
    points=[]
    for condition in conditions:
        sw=next((r for r in all_rows if r['condition']==condition and r['task']=='knowledge_choice' and r['input_language']=='sw' and r['metric']=='content_correct'),None)
        en=next((r for r in all_rows if r['condition']==condition and r['task']=='math' and r['input_language']=='en' and r['metric']=='content_correct'),None)
        if sw and en:points.append((condition,sw,en))
    if points:
        fig,ax=plt.subplots(figsize=(9,7))
        for name,sw,en in points:
            ax.scatter(sw['effect_final_minus_intervened_pp'],en['effect_final_minus_intervened_pp'],marker='o' if sw['component']=='attention' else '^');ax.annotate(name,(sw['effect_final_minus_intervened_pp'],en['effect_final_minus_intervened_pp']),fontsize=6,xytext=(2,3),textcoords='offset points')
        ax.axhline(0,color='gray',linewidth=.7);ax.axvline(0,color='gray',linewidth=.7);ax.set_xlabel('SW knowledge-choice conditional effect (pp)');ax.set_ylabel('EN math-content conditional effect (pp)');ax.set_title('Diagnostic points; consult cluster intervals in CSV');fig.tight_layout();fig.savefig(a.output/'conditional_effect_scatter.png',dpi=160);plt.close(fig)
    print(json.dumps({'matrix_rows':len(all_rows),'conditions':len(conditions)}))
if __name__=='__main__':main()
