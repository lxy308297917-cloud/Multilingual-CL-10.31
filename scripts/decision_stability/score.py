"""Frozen fresh development readouts and independent training-only teacher cache."""
import argparse, datetime, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    for key in ['project','model','data','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=['dev','teacher','holdout'],required=True)
    a=p.parse_args();sys.path.insert(0,str(a.project/'scripts/answer_bias_study'))
    from core import score_labels,score_pairs,softmax,remap
    def native_labels(model,rotations,device,pad):
        result=[]
        for pairs in rotations:
            c=pairs[0][0];prefix=pairs[0][1][:-1]
            assert all(cc==c and t[:-1]==prefix for cc,t in pairs)
            ids=torch.tensor([c+prefix],device=device)
            logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits[0,-1].float()
            result.append(logits[[t[-1] for _,t in pairs]].log_softmax(-1).tolist())
        return result
    assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-')
    assert ',' not in os.environ['CUDA_VISIBLE_DEVICES']
    assert not a.output.exists();a.output.mkdir(parents=True);start=time.time()
    def dump(n,v):(a.output/n).write_text(json.dumps(v,indent=2)+'\n')
    try:
        protocol=json.loads((a.data/'protocol.json').read_text())
        split='train_aux' if a.mode=='teacher' else a.mode
        assert sha(a.data/(split+'.jsonl'))==protocol['files_sha256'][split+'.jsonl']
        dtype=torch.bfloat16 if a.mode=='teacher' else torch.float32
        dump('launch.json',dict(model=str(a.model.resolve()),mode=a.mode,dtype=str(dtype),
             protocol_sha256=sha(a.data/'protocol.json'),code_sha256=sha(__file__),
             weights={v.name:sha(v) for v in a.model.glob('*.safetensors')},
             gpu=os.environ['CUDA_VISIBLE_DEVICES'],started=datetime.datetime.now().astimezone().isoformat()))
        torch.set_num_threads(4);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype,
            attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
        model.config.use_cache=False;pad=model.config.pad_token_id or model.config.eos_token_id
        if isinstance(pad,list):pad=pad[0]
        rows=[json.loads(s) for s in (a.data/(split+'.jsonl')).open()]
        if a.mode=='teacher':rows=[r for r in rows if r['language'] in ['en','sw']]
        predictions=[];tokens=0;forwards=0
        with torch.inference_mode(),(a.output/'predictions.jsonl').open('x') as out:
            # Canonical native Qwen output verifies the memory-saving scorer.
            pairs=rows[0]['rotations'][0]['letter_pairs']
            got=score_labels(model,[pairs],'cuda:0',pad)[0];errors=[]
            for score,(c,t) in zip(got,pairs):
                ids=torch.tensor([(c+t)[:-1]],device='cuda:0')
                h=model.model(input_ids=ids,use_cache=False).last_hidden_state[:,-len(t):]
                lp=model.lm_head(h.float() if dtype==torch.float32 else h).float().log_softmax(-1)[0]
                want=sum(float(lp[i,z]) for i,z in enumerate(t));errors.append(abs(score-want))
            # BF16 head arithmetic differs from scorer's FP32 projection; targets
            # use the latter consistently in training, with a separate oracle.
            dump('scorer_check.json',dict(native_head_errors=errors,dtype=str(dtype)))
            if dtype==torch.float32:assert max(errors)<1e-3,errors
            for i,v in enumerate(rows):
                scorer=native_labels if a.mode=='teacher' else score_labels
                scores=scorer(model,[r['letter_pairs'] for r in v['rotations']],'cuda:0',pad)
                rotations=[]
                for r,ls in zip(v['rotations'],scores):
                    rr=dict(rotation=r['rotation'],order=r['order'],letter_lp=ls)
                    if a.mode!='teacher':
                        cs,ln=score_pairs(model,r['content_pairs'],'cuda:0',pad)
                        rr.update(content_lp=cs,content_tokens=ln,content_mean_lp=[x/n for x,n in zip(cs,ln)])
                        tokens+=sum(len(c)+len(t)-1 for c,t in r['content_pairs']);forwards+=4
                    tokens+=len(r['letter_pairs'][0][0])+len(r['letter_pairs'][0][1])-1;forwards+=1
                    rotations.append(rr)
                pred={k:v[k] for k in ['id','language','gold','split']};pred['rotations']=rotations
                probs=np.mean([remap(softmax(r['letter_lp']),r['order']) for r in rotations],axis=0)
                pred['content_probability']=probs.tolist()
                pred['correct']=int(probs.argmax())==v['gold']
                pred['margin']=float(np.sort(probs)[-1]-np.sort(probs)[-2])
                if a.mode!='teacher':
                    controls=v['position_controls'];control_scores=score_labels(model,[r['letter_pairs'] for r in controls],'cuda:0',pad)
                    pred['position_controls']=[dict(rotation=r['rotation'],label_shift=r['label_shift'],
                        label_to_position=r['label_to_position'],label_to_content=r['label_to_content'],letter_lp=ls)
                        for r,ls in zip(controls,control_scores)]
                    tokens+=sum(len(r['letter_pairs'][0][0])+len(r['letter_pairs'][0][1])-1 for r in controls);forwards+=len(controls)
                predictions.append(pred);out.write(json.dumps(pred)+'\n')
                if (i+1)%20==0:
                    out.flush();dump('progress.json',dict(rows=i+1,total=len(rows),seconds=time.time()-start))
        if a.mode=='teacher':
            groups={}
            for v in predictions:groups.setdefault(v['id'],{})[v['language']]=v
            selected=sorted(k for k,g in groups.items() if len(g)==2 and all(v['correct'] and v['margin']>1e-6 for v in g.values()))
            marginal={lang:np.mean([softmax(r['letter_lp']) for k in selected for r in groups[k][lang]['rotations']],axis=0).tolist() for lang in ['en','sw']} if selected else {}
            dump('teacher.json',dict(selected_ids=selected,selected_count=len(selected),label_marginal=marginal,
                targets={k:{l:g[l]['content_probability'] for l in ['en','sw']} for k,g in groups.items() if k in selected},
                rule=protocol['teacher_rule'],arithmetic='Native BF16 model output head, FP32 four-candidate logsoftmax; same as student'))
        dump('DONE.json',dict(rows=len(rows),seconds=time.time()-start,input_tokens=tokens,forward_sequences=forwards,
             gpu_wall_hours=(time.time()-start)/3600,peak_cuda=torch.cuda.max_memory_allocated(),prediction_sha256=sha(a.output/'predictions.jsonl')))
    except BaseException as e:
        dump('FAILED.json',dict(error=repr(e),seconds=time.time()-start));raise

if __name__=='__main__':main()
