"""CPU recomputation of exported SIB/Global-MMLU statistics and table averages."""
from pathlib import Path
import argparse, csv, hashlib, itertools, json
import numpy as np

LABELS = ['science/technology','travel','politics','sports','health','entertainment','geography']
TARGET = ['Belebele','XNLI','Global-MMLU','SIB-200']
GENERAL = ['GSM8K','IFEval','HumanEval+','EN MMLU','EN Belebele']

def readcsv(p): return list(csv.DictReader(p.open()))
def check(a, b): assert abs(float(a)-float(b)) < 1e-9, (a,b)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--require-complete',action='store_true');args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];q=root/'research/benchmark_5plus5_20261109'
    models=json.loads((root/'configs/benchmark_5plus5_20261109/models.json').read_text())
    deferred=set(json.loads((root/'configs/benchmark_5plus5_20261109/deferred_models.json').read_text()))
    table={r['model']:r for r in readcsv(q/'scores_5plus5.csv')};assert set(table)==set(models)
    diagnostics=json.loads((q/'sib_diagnostics.json').read_text());vectors={};inputs=[]
    legacy_path=root/'research/updates_20261108/aya-4b-generalization-20261107/three-language-scores.csv'
    legacy=readcsv(legacy_path);legacy_count=0
    for lang in ['sw','ar','ru']:
        for task,metric,destination,samples in [('Belebele target','ACC','Belebele',900),('XNLI target','ACC','XNLI',5010),('XL-Sum target','ROUGE-Lsum','XL-Sum',None),('Belebele EN','ACC','EN Belebele',900),('GSM8K','flexible-extract','GSM8K',1319),('IFEval','prompt_level_strict_acc','IFEval',541)]:
            matching=[x for x in legacy if (x['language'],x['task'],x['metric'])==(lang,task,metric)]
            assert len(matching)==1,(lang,task,metric)
            if samples is not None:assert int(matching[0]['n'])==samples
            check(table[lang+'_historical_lora'][destination],matching[0]['final']);legacy_count+=1
    assert legacy_count==18
    inputs.append({'path':str(legacy_path.relative_to(root)),'sha256':hashlib.sha256(legacy_path.read_bytes()).hexdigest()})
    for name,m in models.items():
        p=q/'numeric'/(name+'_sib200.json')
        if p.exists():
            assert name not in deferred
            raw=p.read_bytes();xs=json.loads(raw);assert len(xs)==204
            assert len({x['id'] for x in xs})==204
            conf=np.zeros((7,7),dtype=int);primary={};secondary={}
            for x in xs:
                ll=np.asarray(x['option_loglikelihoods'],float);assert ll.shape==(7,) and np.isfinite(ll).all()
                gold=x['gold'];assert isinstance(gold,int) and 0<=gold<7
                pred=int(np.argmax(ll/np.array([len(t) for t in LABELS])));raw_pred=int(np.argmax(ll))
                assert pred==x['pred'] and raw_pred==x['raw_pred']
                identity=x['input_identity_sha256'];assert len(identity)==64
                primary[x['id']]={'identity':identity,'value':100*int(pred==gold)}
                secondary[x['id']]={'identity':identity,'value':100*int(raw_pred==gold)}
                conf[gold,pred]+=1
            score=100*conf.trace()/204;check(score,table[name]['SIB-200']);d=diagnostics[name]
            check(score,d['primary_acc_norm']);check(np.mean([x['value'] for x in secondary.values()]),d['raw_acc'])
            assert conf.tolist()==d['confusion_matrix_gold_rows_pred_columns']
            counts=conf.sum(1);predcounts=conf.sum(0)
            f1=np.divide(2*np.diag(conf),counts+predcounts,out=np.zeros(7),where=counts+predcounts>0)
            check(100*f1.mean(),d['macro_f1'])
            vectors[name,'SIB-200']=primary;vectors[name,'SIB-200 raw']=secondary
            inputs.append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(raw).hexdigest()})
        p=q/'numeric'/(name+'_global_mmlu.json')
        if p.exists():
            raw=p.read_bytes();v=json.loads(raw);assert v['model']==name and v['language']==m['language']
            assert v['task']=='Global-MMLU' and v['samples']==14042
            assert len(v['item_ids'])==len(set(v['item_ids']))==len(v['input_identity_sha256'])==len(v['values'])==14042
            assert all(x in [0,100] for x in v['values'])
            check(np.mean(v['values']),table[name]['Global-MMLU'])
            subject=[r for r in readcsv(q/'gmmlu_subjects.csv') if r['model']==name]
            assert len(subject)==57 and sum(int(r['samples']) for r in subject)==14042
            check(sum(int(r['correct']) for r in subject)*100/14042,np.mean(v['values']))
            for r in subject:check(int(r['correct'])*100/int(r['samples']),r['acc'])
            vectors[name,'Global-MMLU']={k:{'identity':h,'value':x} for k,h,x in zip(v['item_ids'],v['input_identity_sha256'],v['values'])}
            inputs.append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(raw).hexdigest()})
    pairs=readcsv(q/'paired_changes.csv');expected={(a,b,t) for a,b in itertools.combinations(models,2)
        if models[a]['language']==models[b]['language'] for t in ['SIB-200','SIB-200 raw','Global-MMLU']
        if (a,t) in vectors and (b,t) in vectors}
    observed=set()
    for row in pairs:
        key=row['reference'],row['model'],row['task'];assert key in expected and key not in observed;observed.add(key)
        a,b=vectors[key[0],key[2]],vectors[key[1],key[2]];assert a.keys()==b.keys()
        keys=sorted(a);assert all(a[k]['identity']==b[k]['identity'] for k in keys)
        delta=np.array([b[k]['value']-a[k]['value'] for k in keys],float)
        assert len(keys)==int(row['samples']) and int(row['resamples'])==2000
        rng=np.random.default_rng(42);draws=[np.mean(delta[rng.integers(len(keys),size=len(keys))]) for _ in range(2000)]
        low,high=np.quantile(draws,[.025,.975])
        for field,value in [('delta',delta.mean()),('ci95_low',low),('ci95_high',high)]:check(row[field],value)
    assert observed==expected
    count=0
    for name,row in table.items():
        if name in deferred:assert row['execution_status']=='USER_DEFERRED';continue
        for fields,output in [(TARGET,'TargetACC4'),(GENERAL,'GeneralAvg5')]:
            if all(row[t]!='' for t in fields):check(np.mean([float(row[t]) for t in fields]),row[output])
            else:assert row[output]==''
        count+=sum(row[t]!='' for t in TARGET+GENERAL+['XL-Sum'])
    required=(len(models)-len(deferred))*10
    sib=sum(t=='SIB-200' for n,t in vectors);gm=sum(t=='Global-MMLU' for n,t in vectors)
    if args.require_complete:assert sib==len(models)-len(deferred) and gm==4 and count==required
    report=dict(status='PASS_AVAILABLE_EXPORTED_NUMERIC',all_eligible_cells_present=count==required,
                legacy_primary_cells_verified=legacy_count,sib_models_recomputed=sib,full_gmmlu_models_recomputed=gm,paired_intervals_recomputed=len(pairs),
                present_5plus5_cells=count,required_5plus5_cells=required,resamples=2000,numpy=np.__version__,inputs=inputs,
                scope='SIB option likelihoods and full GM correctness/paired intervals plus descriptive table means only. Other task raw scoring depends on the separate formal/FFT audits; this is not proof of all ten raw task scores.',
                table_sha256={n:hashlib.sha256((q/n).read_bytes()).hexdigest() for n in ['scores_5plus5.csv','paired_changes.csv','gmmlu_subjects.csv']})
    (q/'numeric_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','sib_models_recomputed','full_gmmlu_models_recomputed','paired_intervals_recomputed','present_5plus5_cells','required_5plus5_cells']}))

if __name__=='__main__':main()
