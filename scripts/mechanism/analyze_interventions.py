"""Completed-run effects with duplicate-prompt clustering and interaction limits.

Adds analysis only; never changes frozen generation or original primary scores.
"""
import argparse,collections,csv,json,re
from decimal import Decimal,InvalidOperation
from pathlib import Path
import numpy as np
from build_impact_matrix import impact_rows,cluster_interval


def equivalence_clusters(rows):
    parent={r['id']:r['id'] for r in rows};seen_pair={};seen_prompt={}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for r in rows:
        scope=(r['task'],r['input_language'],r['output_language']);id=r['id']
        if r.get('pair_id') is not None:
            key=scope+(r['pair_id'],)
            if key in seen_pair:join(id,seen_pair[key])
            else:seen_pair[key]=id
        key=scope+(r['prompt_token_sha256'],)
        if key in seen_prompt:join(id,seen_prompt[key])
        else:seen_prompt[key]=id
    return {id:find(id) for id in parent}


def clustered_effects(reference,condition,repeats=2000):
    clusters=equivalence_clusters(reference)
    def copied(rows):return [dict(r,pair_id=clusters[r['id']]) for r in rows]
    result=impact_rows(copied(reference),copied(condition),repeats)
    for r in result:r['cluster_unit']='Transitive equivalence of same pair ID or identical prompt hash within task/input/output-language group'
    return result


def interaction_rows(final,attention,mlp,block,remaining_norms_changed,repeats=2000):
    sets=[{r['id']:r for r in rows} for rows in [final,attention,mlp,block]]
    if any(s.keys()!=sets[0].keys() for s in sets[1:]):raise ValueError('Interaction sample IDs differ')
    clusters=equivalence_clusters(final);groups=collections.defaultdict(list)
    for id,f in sets[0].items():
        a,m,b=[s[id] for s in sets[1:]]
        for row in [a,m,b]:
            if any(row[k]!=f[k] for k in ['task','input_language','output_language','prompt_token_sha256','max_new_tokens']):raise ValueError('Interaction protocols differ')
        for metric in ['content_correct','format_passed','language_correct']:
            values=[r['score'][metric] for r in [f,a,m,b]]
            if all(v is None for v in values):continue
            if any(v is None for v in values):raise ValueError('Interaction metric denominator changed')
            value=float(values[1])+float(values[2])-float(values[3])-float(values[0]);groups[(f['task'],f['input_language'],f['output_language'],metric)].append((value,clusters[id]))
    result=[]
    for key,values in sorted(groups.items()):
        point,low,high,n=cluster_interval([v for v,_ in values],[c for _,c in values],repeats)
        result.append(dict(task=key[0],input_language=key[1],output_language=key[2],metric=key[3],samples=len(values),clusters=n,contrast_pp=point*100,ci95_low_pp=None if low is None else low*100,ci95_high_pp=None if high is None else high*100,formula='Score(restoreAttention)+Score(restoreMLP)-Score(restoreBlock)-Score(Final)',remaining_block_components_changed=remaining_norms_changed,interpretation='Conditional nonadditivity on this cohort' if remaining_norms_changed is False else 'Nonadditivity PLUS unseparated effects of block components outside Attention/MLP; not a pure interaction'))
    return result


def extra_block_components_changed(drift_path,block_names,component_names):
    if drift_path is None:return None,[]
    data=json.loads(Path(drift_path).read_text());observed={alias for r in data['parameters'] for alias in r.get('aliases',[r['parameter']])}
    if not set(block_names).issubset(observed):raise ValueError('Drift audit missing selected block parameters')
    names=set(block_names)-set(component_names);outside=[r for r in data['parameters'] if r['parameter'] in names]
    return any(r['changed_elements']>0 for r in outside),[dict(parameter=r['parameter'],changed_elements=r['changed_elements'],delta_norm=r['delta_norm']) for r in outside]


def generation_diagnostics(rows,probe_rows,eos_id):
    target={r['id']:r for r in probe_rows};groups=collections.defaultdict(list)
    for r in rows:groups[(r['task'],r['input_language'],r['output_language'])].append(r)
    output=[]
    for key,items in sorted(groups.items()):
        generated=[r for r in items if r['score'].get('predicted_option') is None]
        comma_cases=[]
        for r in items:
            if r['task']!='math':continue
            matches=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$',r['raw_response'])
            if not matches:continue
            try:correct=Decimal(matches[-1].replace(',',''))==Decimal(str(target[r['id']]['answer']))
            except InvalidOperation:correct=False
            if correct and not r['score']['content_correct']:comma_cases.append(r['id'])
        output.append(dict(task=key[0],input_language=key[1],output_language=key[2],samples=len(items),generated_samples=len(generated),raw_length_equals_cap=sum(bool(r['generation_hit_cap']) for r in generated),cap_without_eos=sum(bool(r['generation_hit_cap']) and bool(r['generated_token_ids']) and r['generated_token_ids'][-1]!=eos_id for r in generated),empty_outputs=sum(not r['raw_response'].strip() for r in generated),mean_generated_tokens=float(np.mean([r['generated_tokens'] for r in generated])) if generated else None,comma_normalized_math_correct_but_original_false=comma_cases,diagnostic_only_primary_scores_unchanged=True))
    return output


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as stream:
        w=csv.DictWriter(stream,list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--probes',type=Path,required=True);p.add_argument('--parameter-drift',type=Path);p.add_argument('--eos-id',type=int,required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args();done=json.loads((a.run/'DONE.json').read_text());a.output.mkdir(parents=True,exist_ok=False)
    def read(label):
        with (a.run/label/'predictions.jsonl').open() as stream:return [json.loads(x) for x in stream]
    all_predictions={label:read(label) for label in done['conditions']};final=all_predictions['final'];effects=[];diagnostics={}
    with a.probes.open() as stream:probes=[json.loads(x) for x in stream]
    for label,rows in all_predictions.items():
        diagnostics[label]=generation_diagnostics(rows,probes,a.eos_id)
        if label in ('base','final'):continue
        metadata=json.loads((a.run/label/'summary.json').read_text())['metadata'];transaction=metadata['intervention']
        if not transaction['restoration_verified']:raise ValueError('Failed rollback')
        for row in clustered_effects(final,rows,a.bootstrap):
            row.update(condition=label,mode=metadata['mode'],alpha=metadata['alpha'],component=metadata['component'],layers=metadata['layers'],selected_parameters=metadata['parameter_count'],actual_changed_parameters=transaction['changed_unique_elements'],actual_replacement_norm=transaction['actual_replacement_norm']);effects.append(row)
    interaction=[];changed=None;remaining=[]
    required=['restore_L14_attention','restore_L14_mlp','restore_L14_block']
    if all(k in all_predictions for k in required):
        metadata=[json.loads((a.run/k/'summary.json').read_text())['metadata'] for k in required]
        changed,remaining=extra_block_components_changed(a.parameter_drift,metadata[2]['selected_parameter_names'],metadata[0]['selected_parameter_names']+metadata[1]['selected_parameter_names'])
        interaction=interaction_rows(final,*(all_predictions[k] for k in required),changed,a.bootstrap)
    report={'scope':'Frozen diagnostic probes only; primary scores unchanged; not official benchmark or broad causal attribution','source_protocol':json.loads((a.run/'protocol.json').read_text()),'cluster_method':'Pair ID or exact prompt hash equivalence, within task/input/output groups; point estimates retain predetermined row weights','intervals':'Paired cluster bootstrap percentile95%, exploratory; no multiplicity correction or representativeness guarantee; degenerate observed CI is not population equality','effect_sign':'Final minus intervention: positive means intervention lowers this score, negative means it improves score. Inject/half/random are contrasts, not all literal restores.','component_assignment':'Uses exact selected parameter-name sets from intervention metadata and unique-parameter drift audit; extra block components reported separately.','block_components_outside_attention_mlp':remaining,'unseparated_block_changes':changed,'rows':effects,'interaction':interaction,'functional_separation_established':False}
    (a.output/'conditional_effects.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');write_csv(a.output/'conditional_effects.csv',effects);write_csv(a.output/'interaction_contrasts.csv',interaction);(a.output/'generation_diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=sorted({(r['task'],r['input_language'],r['output_language'],r['metric']) for r in effects});labels=[c for c in done['conditions'] if c not in ('base','final')];array=np.full((len(labels),len(keys)),np.nan)
    for r in effects:array[labels.index(r['condition']),keys.index((r['task'],r['input_language'],r['output_language'],r['metric']))]=r['effect_final_minus_intervened_pp']
    fig,ax=plt.subplots(figsize=(max(13,len(keys)*.45),8));bound=max(1.,float(np.nanmax(abs(array))));im=ax.imshow(array,cmap='coolwarm',vmin=-bound,vmax=bound,aspect='auto');ax.set_yticks(range(len(labels)),labels,fontsize=7);ax.set_xticks(range(len(keys)),['/'.join(k) for k in keys],rotation=90,fontsize=6);fig.colorbar(im,ax=ax,label='Final minus intervention (pp); see intervals');fig.tight_layout();fig.savefig(a.output/'conditional_effect_heatmap.png',dpi=160);fig.savefig(a.output/'conditional_effect_heatmap.svg');plt.close(fig)
    points=[]
    for label in labels:
        sw=next((r for r in effects if r['condition']==label and r['task']=='knowledge_choice' and r['input_language']=='sw' and r['metric']=='content_correct'),None);en=next((r for r in effects if r['condition']==label and r['task']=='math' and r['input_language']=='en' and r['metric']=='content_correct'),None)
        if sw and en:points.append((label,sw,en))
    if points:
        fig,ax=plt.subplots(figsize=(10,8))
        for label,x,y in points:
            xv=x['effect_final_minus_intervened_pp'];yv=y['effect_final_minus_intervened_pp'];xe=None if x['ci95_low_pp'] is None else [[max(0,xv-x['ci95_low_pp'])],[max(0,x['ci95_high_pp']-xv)]];ye=None if y['ci95_low_pp'] is None else [[max(0,yv-y['ci95_low_pp'])],[max(0,y['ci95_high_pp']-yv)]]
            ax.errorbar(xv,yv,xerr=xe,yerr=ye,fmt='o' if x['component']=='attention' else '^',alpha=.55,capsize=2);ax.annotate(label,(xv,yv),fontsize=6,xytext=(2,2),textcoords='offset points')
        ax.axhline(0,color='gray',linewidth=.7);ax.axvline(0,color='gray',linewidth=.7);ax.set_xlabel('SW knowledge-choice conditional effect (pp)');ax.set_ylabel('EN math-content conditional effect (pp)');ax.set_title('Exploratory paired-cluster intervals; no multiplicity correction');fig.tight_layout();fig.savefig(a.output/'conditional_effect_scatter.png',dpi=160);fig.savefig(a.output/'conditional_effect_scatter.svg');plt.close(fig)
    print(json.dumps({'effect_rows':len(effects),'interaction_rows':len(interaction),'conditions':len(labels),'unseparated_block_changes':changed}))
if __name__=='__main__':main()
