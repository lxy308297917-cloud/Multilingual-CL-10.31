"""Aggregate all frozen contrasts; no prompt/model/calibration selection."""
from pathlib import Path
import argparse, collections, datetime, hashlib, itertools, json
import numpy as np
from core import softmax, remap, pride_prior, corrected


def ci(values, clusters):
    values=np.asarray(values,dtype=float); keys=sorted(set(clusters));m={k:i for i,k in enumerate(keys)}
    sums=np.zeros(len(keys));counts=np.zeros(len(keys))
    for x,k in zip(values,clusters):sums[m[k]]+=x;counts[m[k]]+=1
    rng=np.random.default_rng(42);idx=rng.integers(0,len(keys),(2000,len(keys)))
    boots=sums[idx].sum(axis=1)/counts[idx].sum(axis=1)
    return dict(mean=float(values.mean()),low=float(np.quantile(boots,.025)),high=float(np.quantile(boots,.975)),n=len(values),clusters=len(keys))


def loadrows(folder):
    assert (folder/'DONE.json').exists(),folder
    return [json.loads(s) for s in (folder/'predictions.jsonl').open()]


def row_metrics(row, prior, blank):
    rotations=row['rotations'];gold=row['gold']
    ls=np.array([r['letter_lp'] for r in rotations]);lp=softmax(ls)
    cms=np.array([r['content_mean_lp'] for r in rotations]);crs=np.array([r['content_lp'] for r in rotations])
    letter_probs=np.array([remap(p,r['order']) for p,r in zip(lp,rotations)])
    content_probs=np.array([remap(p,r['order']) for p,r in zip(softmax(cms),rotations)])
    label_preds=ls.argmax(axis=1);content_preds=[r['order'][int(x.argmax())] for r,x in zip(rotations,cms)]
    letter_preds=[r['order'][int(x)] for r,x in zip(rotations,label_preds)]
    ensemble=letter_probs.mean(axis=0); ensemble_top=np.sort(ensemble)
    tied=np.isclose(ensemble,ensemble.max(),rtol=0,atol=1e-12)
    methods={'raw_letter':letter_preds[0], 'position_average':int(ensemble.argmax()),
             'content_mean':content_preds[0], 'content_raw':rotations[0]['order'][int(crs[0].argmax())],
             'content_position_average':int(content_probs.mean(axis=0).argmax()),
             'pride':int(corrected(ls[0],prior).argmax()), 'blank_calibration':int(corrected(ls[0],blank).argmax())}
    random=[]
    for perm in itertools.permutations(range(4)):
        if perm==(0,1,2,3):continue
        random.append(int(corrected(ls[0],np.asarray(prior)[list(perm)]).argmax()))
    methods['random_prior_mean_correct']=float(np.mean([v==gold for v in random]))
    # Rotation zero always identity, so direct prior-corrected argmax is content ID.
    margins=[]
    for p in letter_probs:
        margins.append(float(np.log(max(p[gold],1e-300))-np.log(max(np.delete(p,gold).max(),1e-300))))
    return dict(id=row['id'],cluster=row['cluster'],subject=row['subject'],gold=gold,
                predictions={k:v for k,v in methods.items() if k!='random_prior_mean_correct'},
                correct={k:int(v==gold) for k,v in methods.items() if k!='random_prior_mean_correct'},
                random_prior_correct=[int(v==gold) for v in random],random_prior_mean_correct=methods['random_prior_mean_correct'],
                rotation_correct=[int(v==gold) for v in letter_preds],content_rotation_correct=[int(v==gold) for v in content_preds],
                label_predictions=label_preds.tolist(),content_predictions=content_preds,letter_content_predictions=letter_preds,
                letter_unanimity=int(len(set(letter_preds))==1),content_unanimity=int(len(set(content_preds))==1),
                content_pair_agreement=sum(a==b for a,b in itertools.combinations(content_preds,2))/6,
                letter_pair_agreement=sum(a==b for a,b in itertools.combinations(letter_preds,2))/6,
                label_unanimity=int(len(set(label_preds.tolist()))==1),
                rotation_gold_probability=[float(p[gold]) for p in letter_probs],
                rotation_gold_margin=margins,
                letter_entropy=float(np.mean(-(lp*np.log(np.maximum(lp,1e-300))).sum(axis=1))),
                content_entropy=float(np.mean(-(softmax(cms)*np.log(np.maximum(softmax(cms),1e-300))).sum(axis=1))),
                mean_content_tokens=float(np.array([r['content_tokens'] for r in rotations]).mean()),
                content_length_range=max(rotations[0]['content_tokens'])-min(rotations[0]['content_tokens']),
                position_average_probability_gap=float(ensemble_top[-1]-ensemble_top[-2]),
                position_average_tie_count=int(tied.sum()),position_average_fractional_tie_correct=float(tied[gold]/tied.sum()),
                raw_letter_score_gap=float(np.sort(ls[0])[-1]-np.sort(ls[0])[-2]),
                top_label_ties=[int(np.sum(x==x.max())) for x in ls])


def summary(rows,gate):
    clusters=[v['cluster'] for v in rows];n=len(rows)
    methods=list(rows[0]['correct'])
    if not gate:methods=[m for m in methods if m not in ['pride','blank_calibration']]
    metrics={m:ci([v['correct'][m] for v in rows],clusters) for m in methods}
    paired={m+'_minus_raw':ci([v['correct'][m]-v['correct']['raw_letter'] for v in rows],clusters) for m in methods if m!='raw_letter'}
    label_counts=np.bincount(np.array([v['label_predictions'] for v in rows]).ravel(),minlength=4)
    result=dict(n=n,calibration_gate=gate,metrics=metrics,paired=paired,
                label_counts_all_rotations=label_counts.tolist(),label_share_all_rotations=(label_counts/(4*n)).tolist(),
                raw_predicted_counts=np.bincount([v['predictions']['raw_letter'] for v in rows],minlength=4).tolist(),
                gold_counts=np.bincount([v['gold'] for v in rows],minlength=4).tolist(),
                letter_unanimity=float(np.mean([v['letter_unanimity'] for v in rows])),
                content_unanimity=float(np.mean([v['content_unanimity'] for v in rows])),
                label_unanimity=float(np.mean([v['label_unanimity'] for v in rows])),
                letter_pair_agreement=float(np.mean([v['letter_pair_agreement'] for v in rows])),
                content_pair_agreement=float(np.mean([v['content_pair_agreement'] for v in rows])),
                letter_entropy=float(np.mean([v['letter_entropy'] for v in rows])),
                content_entropy=float(np.mean([v['content_entropy'] for v in rows])),
                rotation_mean_accuracy=float(np.mean([v['rotation_correct'] for v in rows])),
                rotation_accuracies=np.mean([v['rotation_correct'] for v in rows],axis=0).tolist(),
                rotation_content_mean_accuracy=float(np.mean([v['content_rotation_correct'] for v in rows])),
                by_gold_position={str(i):float(np.mean([v['rotation_correct'][r] for v in rows for r in range(4) if (v['gold']-r)%4==i])) for i in range(4)},
                mean_gold_probability=float(np.mean([v['rotation_gold_probability'] for v in rows])),
                mean_gold_margin=float(np.mean([v['rotation_gold_margin'] for v in rows])),
                mean_content_tokens=float(np.mean([v['mean_content_tokens'] for v in rows])),
                exact_top_label_ties=sum(sum(z>1 for z in v['top_label_ties']) for v in rows),
                position_average_top_ties=sum(v['position_average_tie_count']>1 for v in rows),
                position_average_probability_gap_below1e6=sum(v['position_average_probability_gap']<=1e-6 for v in rows),
                position_average_fractional_tie_accuracy=ci([v['position_average_fractional_tie_correct'] for v in rows],clusters),
                raw_letter_score_gap_below1e3=sum(v['raw_letter_score_gap']<=1e-3 for v in rows),
                length_strata={name:dict(n=len(vs),raw=float(np.mean([v['correct']['raw_letter'] for v in vs])),content_mean=float(np.mean([v['correct']['content_mean'] for v in vs])),content_raw=float(np.mean([v['correct']['content_raw'] for v in vs]))) for name,vs in [(name,[v for v in rows if lo<=v['content_length_range']<hi]) for name,lo,hi in [('equal',0,1),('range1to5',1,6),('range6plus',6,1e9)]] if vs})
    if gate:
        result['random_prior_control']=dict(n_permutations=23,accuracies=np.mean([v['random_prior_correct'] for v in rows],axis=0).tolist(),
                 mean=ci([v['random_prior_mean_correct'] for v in rows],clusters),
                 pride_minus_random_mean=ci([v['correct']['pride']-v['random_prior_mean_correct'] for v in rows],clusters))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    science=json.loads(a.protocol.read_text());plan={'models':dict.fromkeys(science['model_names']), 'protocol_sha256':science['protocol_sha256']};run=a.run_root
    assert not a.output.exists();a.output.mkdir(parents=True)
    allrows={};summaries={};priors={};receipts={}
    for label in plan['models']:
        folder=run/label;raw=loadrows(folder);receipts[label]=json.loads((folder/'DONE.json').read_text())
        assert receipts[label]['protocol_sha256']==plan['protocol_sha256']
        blankrows=json.loads((folder/'content_free_predictions.json').read_text())
        blank={t:softmax([v['letter_lp'] for v in blankrows if v['task']==t]).mean(axis=0) for t in ['reading_exposed','knowledge_dev']}
        prior={l:pride_prior([[r['letter_lp'] for r in v['rotations']] for v in raw if v['split']=='calibration' and v['language']==l]) for l in ['en','sw','ha']}
        priors[label]=dict(knowledge_calibration={l:p.tolist() for l,p in prior.items()},content_free={t:p.tolist() for t,p in blank.items()})
        groups=collections.defaultdict(list)
        for v in raw:
            if v['split']=='calibration':continue
            groups[v['task']+'|'+v['language']].append(row_metrics(v,prior[v['language']],blank[v['task']]))
        allrows[label]=groups;summaries[label]={}
        for g,rows in groups.items():
            count=np.bincount(np.array([v['label_predictions'] for v in rows]).ravel(),minlength=4)
            gate=bool(count.max()/count.sum()>=.40 and np.mean([v['letter_unanimity'] for v in rows])<=.80)
            summaries[label][g]=summary(rows,gate)
    # Exact identity across all models and three parallel languages.
    ref=next(iter(allrows.values()))
    for model,groups in allrows.items():
        for g,rows in groups.items():
            assert [(v['id'],v['gold'],v['cluster']) for v in rows]==[(v['id'],v['gold'],v['cluster']) for v in ref[g]]
    cross={};subject={};difficulty={}
    for label,groups in allrows.items():
        cross[label]={};subject[label]={};difficulty[label]={}
        for task in ['reading_exposed','knowledge_dev']:
            maps={l:{v['id']:v for v in groups[task+'|'+l]} for l in ['en','sw','ha']};ids=sorted(maps['en'])
            assert all(set(x)==set(ids) for x in maps.values())
            cross[label][task]={}
            for method in ['raw_letter','position_average','content_mean','content_position_average']:
                patterns=collections.Counter();agreement=0
                for sid in ids:
                    vals=[maps[l][sid] for l in ['en','sw','ha']]
                    assert len(set(v['gold'] for v in vals))==1
                    patterns[''.join(str(v['correct'][method]) for v in vals)]+=1
                    agreement+=len(set(v['predictions'][method] for v in vals))==1
                cross[label][task][method]=dict(n=len(ids),correctness_patterns=dict(patterns),all_language_content_id_agreement=agreement/len(ids))
        for g,rows in groups.items():
            subject[label][g]={}
            for s in sorted(set(v['subject'] for v in rows)):
                vs=[v for v in rows if v['subject']==s]
                subject[label][g][s]=dict(n=len(vs),raw=float(np.mean([v['correct']['raw_letter'] for v in vs])),position_average=float(np.mean([v['correct']['position_average'] for v in vs])),content_mean=float(np.mean([v['correct']['content_mean'] for v in vs])))
            task=g.split('|')[0];base_en={v['id']:v['correct']['position_average'] for v in allrows['base'][task+'|en']}
            difficulty[label][g]={str(k):dict(n=len(vs),raw=float(np.mean([v['correct']['raw_letter'] for v in vs])),position_average=float(np.mean([v['correct']['position_average'] for v in vs])),label_unanimity=float(np.mean([v['label_unanimity'] for v in vs]))) for k,vs in [(k,[v for v in rows if base_en[v['id']]==k]) for k in [0,1]] if vs}
    contrasts={}
    for before,after in [('base','sw_fft'),('base','ha_only'),('sw_fft','sw_ha_fft'),('sw_sst_norefresh','sw_ha_sst_norefresh')]:
        key=before+'__to__'+after;contrasts[key]={}
        for g,left in allrows[before].items():
            right=allrows[after][g];clusters=[v['cluster'] for v in left];methods=list(left[0]['correct'])
            if not (summaries[before][g]['calibration_gate'] and summaries[after][g]['calibration_gate']):methods=[m for m in methods if m not in ['pride','blank_calibration']]
            vals={m:ci([b['correct'][m]-a['correct'][m] for a,b in zip(left,right)],clusters) for m in methods}
            vals['position_average_change_minus_raw_change']=ci([(b['correct']['position_average']-a['correct']['position_average'])-(b['correct']['raw_letter']-a['correct']['raw_letter']) for a,b in zip(left,right)],clusters)
            vals['transitions']={m:dict(collections.Counter(str(a['correct'][m])+str(b['correct'][m]) for a,b in zip(left,right))) for m in methods}
            vals['decision_changed_fraction']={m:float(np.mean([a['predictions'][m]!=b['predictions'][m] for a,b in zip(left,right)])) for m in methods}
            contrasts[key][g]=vals
    result=dict(status='COMPLETE',finished_at=datetime.datetime.now().astimezone().isoformat(),protocol_sha256=plan['protocol_sha256'],summaries=summaries,contrasts=contrasts,cross_language=cross,subject=subject,difficulty=difficulty,priors=priors,compute={k:dict(seconds=v['seconds'],gpu_wall_hours=v['gpu_wall_hours'],peak_cuda_allocated=v['peak_cuda_allocated'],items=v['items'],option_scores=v['option_scores']) for k,v in receipts.items()},total_gpu_wall_hours=sum(v['gpu_wall_hours'] for v in receipts.values()),notes=['Diagnostic/exposed cohorts, not a new formal benchmark or training method.','All intervals conditional on fixed seed42 checkpoints; no training-seed inference or multiple-comparison correction.','Mean token content scores are a distinct cloze protocol, not a certified semantic-ability estimator.','Random control uses all23 nonidentity prior permutations, not best permutation.'])
    (a.output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    # Local only: contains no raw text but complete per-ID outcomes for reproduction.
    with (a.output/'derived_per_item.jsonl').open('x') as f:
        for model,groups in allrows.items():
            for g,rows in groups.items():
                for row in rows:f.write(json.dumps(dict(model=model,group=g,**row))+'\n')
    print(json.dumps(dict(models=len(allrows),protocol=result['protocol_sha256'],total_gpu_wall_hours=result['total_gpu_wall_hours'])))

if __name__=='__main__':main()
