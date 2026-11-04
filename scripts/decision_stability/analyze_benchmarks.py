"""Paired fixed-checkpoint reporting; primary tasks keep native protocols."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np

def interval(sums,counts,clusters,repeats=2000):
    keys=sorted(set(clusters));index={k:i for i,k in enumerate(keys)}
    num=np.zeros(len(keys));den=np.zeros(len(keys))
    for s,c,k in zip(sums,counts,clusters):num[index[k]]+=s;den[index[k]]+=c
    rng=np.random.default_rng(42);values=[]
    for start in range(0,repeats,100):
        draw=rng.integers(0,len(keys),(min(100,repeats-start),len(keys)))
        values.extend((num[draw].sum(1)/den[draw].sum(1)).tolist())
    return dict(delta=float(num.sum()/den.sum()),low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),clusters=len(keys),observations=int(den.sum()),repeats=repeats)

def group_name(name):
    if name.startswith('mmlu_'):return 'EN MMLU'
    if name.startswith('global_mmlu_full_sw_'):return 'SW Global-MMLU'
    if name.startswith('global_mmlu_full_ha_'):return 'HA Global-MMLU'
    if name.startswith('belebele_eng_'):return 'EN Belebele'
    if name.startswith('belebele_swh_'):return 'SW Belebele'
    if name.startswith('belebele_hau_'):return 'HA Belebele'
    if name in ['sft_v1_gsm8k','sft_v1_ifeval','sft_v1_mgsm_native_cot_sw','sft_v1_xifeval_sw']:return name
    return None

def chrf_cached_interval(left,right,refs,repeats=1000):
    from sacrebleu.metrics import CHRF
    scorer=CHRF(char_order=6,word_order=2,beta=2,lowercase=False,whitespace=False,eps_smoothing=False)
    a=np.asarray(scorer._extract_corpus_statistics(left,[refs]),dtype=np.int64)
    b=np.asarray(scorer._extract_corpus_statistics(right,[refs]),dtype=np.int64)
    def score(x):return scorer._compute_score_from_stats(x.sum(0).tolist()).score
    # Validate summed sufficient statistics against the public corpus scorer,
    # including repeated samples as used by the paired bootstrap.
    rng=np.random.default_rng(42)
    errors=[]
    for ids in [np.arange(len(refs)),rng.integers(0,len(refs),len(refs)),rng.integers(0,len(refs),min(31,len(refs)))]:
        for text,stats in [(left,a),(right,b)]:
            exact=scorer.corpus_score([text[i] for i in ids],[[refs[i] for i in ids]]).score
            errors.append(abs(score(stats[ids])-exact))
    assert max(errors)<1e-10,errors
    rng=np.random.default_rng(42);d=[]
    for _ in range(repeats):
        ids=rng.integers(0,len(refs),len(refs));d.append(score(b[ids])-score(a[ids]))
    return dict(delta=score(b)-score(a),low=float(np.quantile(d,.025)),high=float(np.quantile(d,.975)),repeats=repeats,n=len(refs),cached_statistics_oracle_max_error=max(errors))

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    spec=json.loads(a.inputs.read_text());assert not a.output.exists();a.output.mkdir(parents=True)
    allrows={};sources={};scores={};mt={}
    def sha(f):
        h=hashlib.sha256()
        with Path(f).open('rb') as stream:
            for data in iter(lambda:stream.read(8*1024**2),b''):h.update(data)
        return h.hexdigest()
    for model,files in spec.items():
        groups=collections.defaultdict(dict);expected={};mt[model]={}
        for f in files['results']:
            f=Path(f);summary=json.loads(f.read_text());sources[str(f)]=sha(f)
            for name,res in summary['results'].items():
                group=group_name(name)
                if group is None:continue
                samples=list(f.parent.glob(f'samples_{name}_*.jsonl'))
                if not samples:continue
                assert len(samples)==1;sources[str(samples[0])]=sha(samples[0])
                rows=[json.loads(line) for line in samples[0].open()]
                for row in rows:
                    key=(name,str(row['doc_id']),row['filter']);assert key not in groups[group]
                    groups[group][key]=row
                for metric in rows[0]['metrics']:
                    for filt in {row['filter'] for row in rows}:
                        values=[z for row in rows if row['filter']==filt for z in (row[metric] if isinstance(row[metric],list) else [row[metric]])]
                        assert all(z in [0,1] for z in values)
                        assert abs(np.mean(values)-res[metric+','+filt])<1e-10
        allrows[model]=groups;scores[model]={}
        for group,rows in groups.items():
            for filt in sorted(set(k[2] for k in rows)):
                subset=[v for k,v in sorted(rows.items()) if k[2]==filt]
                for metric in subset[0]['metrics']:
                    values=[z for row in subset for z in (row[metric] if isinstance(row[metric],list) else [row[metric]])]
                    scores[model][group+'|'+metric+'|'+filt]=dict(score=float(np.mean(values)),n=len(values),prompts=len(subset))
            n=len({(k[0],k[1]) for k in rows})
            expected_n={'EN Belebele':900,'SW Belebele':900,'HA Belebele':900,'EN MMLU':14042,'SW Global-MMLU':14042,'HA Global-MMLU':14042,'sft_v1_gsm8k':1319,'sft_v1_ifeval':541,'sft_v1_mgsm_native_cot_sw':250,'sft_v1_xifeval_sw':429}[group]
            assert n==expected_n,(model,group,n)
        assert len(groups)==10,(model,list(groups))
        for direction,run in files['translation'].items():
            run=Path(run);manifest=json.loads((run/'manifest.json').read_text());rows=[json.loads(s) for s in (run/'predictions.jsonl').open()]
            assert len(rows)==1012 and len({r['id'] for r in rows})==1012
            assert [r['id'] for r in rows]==manifest['sample_ids']
            mt[model][direction]={r['id']:r for r in rows};sources[str(run/'predictions.jsonl')]=sha(run/'predictions.jsonl')
            score=json.loads((run.parent/'score-v1/summary.json').read_text()) if (run.parent/'score-v1/summary.json').exists() else json.loads((run/'score-v1/summary.json').read_text())
            assert score['samples']==1012 and score['run_status']=='VERIFIED_COMPLETE'
            assert score['source_manifest_sha256']==sha(run/'manifest.json')
            assert score['predictions_sha256']==sha(run/'predictions.jsonl')
            for metric,value in score['metrics'].items():scores[model][direction+'|'+metric]=dict(score=value,n=1012,scale='0..100')
    contrasts={};identities={}
    pairs=[('W1','A'),('W1','B'),('W1','C'),('W1','D'),('A','B'),('A','C'),('A','D'),('B','C'),('D','C')]
    for before,after in pairs:
        result={};identity={}
        assert allrows[before].keys()==allrows[after].keys()
        for group,left in allrows[before].items():
            right=allrows[after][group];assert left.keys()==right.keys()
            for k,row in left.items():
                assert row['doc']==right[k]['doc'] and row['arguments']==right[k]['arguments'],(before,after,group,k,'doc/prompt mismatch')
            identity[group]=dict(rows=len(left),all_doc_and_prompt_exact=True)
            for filt in sorted({k[2] for k in left}):
                keys=sorted(k for k in left if k[2]==filt);rows=[left[k] for k in keys]
                clusters=[str(v['doc']['link']) if 'Belebele' in group else k[0]+'|'+k[1] for k,v in zip(keys,rows)]
                for metric in rows[0]['metrics']:
                    sums=[];counts=[]
                    for k in keys:
                        x=left[k][metric];y=right[k][metric];x=x if isinstance(x,list) else [x];y=y if isinstance(y,list) else [y]
                        assert len(x)==len(y);sums.append(sum(y)-sum(x));counts.append(len(x))
                    result[group+'|'+metric+'|'+filt]=interval(sums,counts,clusters)
        for direction in mt[before]:
            l=mt[before][direction];rr=mt[after][direction];assert l.keys()==rr.keys();ids=sorted(l)
            for k in ids:
                assert l[k]['reference']==rr[k]['reference'] and l[k]['prompt_token_sha256']==rr[k]['prompt_token_sha256'],(direction,k,'translation identity')
            result[direction+'|chrF++']=chrf_cached_interval([l[k]['prediction'] for k in ids],[rr[k]['prediction'] for k in ids],[l[k]['reference'] for k in ids])
        contrasts[after+'_minus_'+before]=result;identities[after+'_minus_'+before]=identity
    payload=dict(scores=scores,contrasts=contrasts,identity_checks=identities,sources_sha256=sources,notes=['Fixed100 checkpoints; original400-step scheduler.','Historically exposed official benchmarks; supplementary reporting, no test-based hyperparameter/checkpoint choice.','Belebele CI clusters original passages, instruction metrics resample prompts and sum instruction counts.','Other ACC metrics use paired original questions; translation recomputes corpus chrF++ from summed ngram statistics, never averages sentence scores.','2000 ACC and1000 corpus-MT bootstrap draws seed42; no multiple-testing correction; one training seed.'])
    (a.output/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Fixed100: all native formal benchmark metrics','','| Task and metric | n | W1 | A SFT | B marginal | C content | D Replay |','|---|---:|---:|---:|---:|---:|---:|']
    for key,row in scores['W1'].items():
        def show(m):
            value=scores[m][key];return f"{value['score']*(1 if value.get('scale')=='0..100' else 100):.2f}"
        lines.append('| '+' | '.join([key.replace('|',' / '),str(row['n'])]+[show(m) for m in ['W1','A','B','C','D']])+' |')
    lines+=['','Percent for ACC and native0–100 for corpus translation scores. All metrics retained. See analysis.json for paired intervals and exact identity checks.']
    (a.output/'table.md').write_text('\n'.join(lines)+'\n');print('Complete',len(scores['W1']),'metrics',len(contrasts),'contrasts')
if __name__=='__main__':main()
