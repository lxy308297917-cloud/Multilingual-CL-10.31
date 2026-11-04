"""Semantic-ID paired comparisons; all frozen readouts retained."""
import argparse,collections,json,sys
from pathlib import Path
import numpy as np

def ci(x):
    x=np.asarray(x,dtype=float);rng=np.random.default_rng(42)
    means=x[rng.integers(0,len(x),(4000,len(x)))].mean(1)
    return dict(n=len(x),mean=float(x.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.project/'scripts/answer_bias_study'));from core import softmax,remap
    assert not a.output.exists();a.output.mkdir(parents=True)
    inputs=json.loads(a.inputs.read_text());allrows={};summaries={};receipts={}
    for label,path in inputs.items():
        folder=Path(path);receipts[label]=json.loads((folder/'DONE.json').read_text());data=[json.loads(s) for s in (folder/'predictions.jsonl').open()]
        groups=collections.defaultdict(list)
        for row in data:
            rotations=row['rotations'];gold=row['gold'];letter=np.asarray([remap(softmax(r['letter_lp']),r['order']) for r in rotations])
            cms=np.asarray([remap(softmax(r['content_mean_lp']),r['order']) for r in rotations])
            pred={'raw_letter':int(np.argmax(rotations[0]['letter_lp'])),'rotation_content_average':int(letter.mean(0).argmax()),
                'content_mean':int(np.argmax(rotations[0]['content_mean_lp'])),'content_raw':int(np.argmax(rotations[0]['content_lp'])),
                'content_rotation_average':int(cms.mean(0).argmax())}
            controls=[dict(letter_lp=r['letter_lp'],label_to_position=list(range(4)),label_to_content=r['order']) for r in rotations]+row['position_controls']
            assert len(controls)==16
            content=[remap(softmax(r['letter_lp']),r['label_to_content']) for r in controls]
            pred['crossed16_content_average']=int(np.mean(content,axis=0).argmax())
            tokenpred=[int(np.argmax(r['letter_lp'])) for r in controls]
            positions=[r['label_to_position'][i] for r,i in zip(controls,tokenpred)]
            contentpred=[r['order'][int(np.argmax(r['letter_lp']))] for r in rotations]
            groups[row['language']].append(dict(id=row['id'],gold=gold,predictions=pred,correct={m:int(v==gold) for m,v in pred.items()},
                tokens=tokenpred,positions=positions,content_unanimity=int(len(set(contentpred))==1),
                gold_probability=float(letter[:,gold].mean()),margin=float(np.log(max(letter.mean(0)[gold],1e-30))-np.log(max(np.delete(letter.mean(0),gold).max(),1e-30))),
                rotation_correct=[int(v==gold) for v in contentpred]))
        allrows[label]={l:sorted(v,key=lambda r:r['id']) for l,v in groups.items()};summaries[label]={}
        for lang,rows in allrows[label].items():
            summaries[label][lang]=dict(metrics={m:ci([v['correct'][m] for v in rows]) for m in rows[0]['correct']},
                token_counts_crossed16=np.bincount(np.array([r['tokens'] for r in rows]).ravel(),minlength=4).tolist(),
                physical_position_counts_crossed16=np.bincount(np.array([r['positions'] for r in rows]).ravel(),minlength=4).tolist(),
                content_unanimity=ci([r['content_unanimity'] for r in rows]),gold_probability=ci([r['gold_probability'] for r in rows]),
                gold_margin=ci([r['margin'] for r in rows]),rotation_mean_accuracy=ci([np.mean(r['rotation_correct']) for r in rows]))
    contrasts={}
    labels=list(allrows)
    for left in labels:
        for right in labels:
            if left==right:continue
            key=right+'_minus_'+left;contrasts[key]={}
            for lang,lrows in allrows[left].items():
                rrows=allrows[right][lang];assert [(r['id'],r['gold']) for r in lrows]==[(r['id'],r['gold']) for r in rrows]
                contrasts[key][lang]={m:ci([b['correct'][m]-c['correct'][m] for c,b in zip(lrows,rrows)]) for m in lrows[0]['correct']}
    result=dict(summaries=summaries,contrasts=contrasts,receipts=receipts,notes=['4000 paired semantic-ID bootstrap draws seed42; no multiple-testing adjustment.','One training seed; no claim of training-seed robustness.','Science-QA is independent diagnosis, not a replacement for target benchmarks.','All readouts are reported; no best-prompt or best-checkpoint selection.'])
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'per_item.json').write_text(json.dumps(allrows)+'\n')
    lines=['# Fresh parallel science-QA readouts','', '| Model | Language | n | Letter ACC | Four-rotation ACC | Content mean ACC | Content raw ACC | Crossed16 ACC |','|---|---|---:|---:|---:|---:|---:|---:|']
    for model,langs in summaries.items():
        for lang,v in langs.items():
            m=v['metrics'];lines.append('| '+ ' | '.join([model,lang,str(m['raw_letter']['n'])]+[f"{100*m[k]['mean']:.2f}" for k in ['raw_letter','rotation_content_average','content_mean','content_raw','crossed16_content_average']])+' |')
    lines+=['','Percent ACC. Independent fresh science-QA; not official Belebele/MMLU. See analysis.json for intervals, paired contrasts and position/token controls.']
    (a.output/'summary.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
