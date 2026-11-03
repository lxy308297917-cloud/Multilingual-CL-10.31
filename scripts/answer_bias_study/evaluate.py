"""One frozen model: rotations, letter/content likelihood, independent blank priors."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, time
import torch
from transformers import AutoModelForCausalLM
from core import score_pairs, score_labels


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for n in ['model','data','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--label',required=True)
    a=p.parse_args(); assert not a.output.exists(); a.output.mkdir(parents=True);start=time.time()
    def dump(n,v):(a.output/n).write_text(json.dumps(v,indent=2)+'\n')
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
        protocol=json.loads((a.data/'protocol.json').read_text())
        for n,h in protocol['files_sha256'].items(): assert sha(a.data/n)==h,n
        weights=sorted(a.model.glob('*.safetensors'));assert weights
        dump('launch.json',dict(label=a.label,model=str(a.model.resolve()),
            weight_sha256={f.name:sha(f) for f in weights},protocol_sha256=sha(a.data/'protocol.json'),
            code_sha256={f.name:sha(f) for f in [Path(__file__),Path(__file__).with_name('core.py')]},
            started_at=datetime.datetime.now().astimezone().isoformat(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],arithmetic=protocol['arithmetic']))
        torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32,
            attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
        model.config.use_cache=False; pad=model.config.pad_token_id or model.config.eos_token_id
        if isinstance(pad,list):pad=pad[0]
        rows=[json.loads(s) for s in (a.data/'items.jsonl').open()]
        # Real model numeric gate before study: canonical full logits on a
        # calibration question, against memory-saving backbone+head scores.
        cal=next(v for v in rows if v['split']=='calibration')
        pairs=cal['rotations'][0]['letter_pairs']; checks=[]
        with torch.inference_mode():
            got=score_labels(model,[pairs],'cuda:0',pad)[0]
            for val,(c,t) in zip(got,pairs):
                ids=torch.tensor([(c+t)[:-1]],device='cuda:0')
                lp=model(input_ids=ids,logits_to_keep=len(t)).logits[0].float().log_softmax(-1)
                want=sum(float(lp[i,z]) for i,z in enumerate(t));checks.append(abs(val-want))
            cp=cal['rotations'][0]['content_pairs']
            got,_=score_pairs(model,cp,'cuda:0',pad)
            for val,(c,t) in zip(got,cp):
                ids=torch.tensor([(c+t)[:-1]],device='cuda:0')
                lp=model(input_ids=ids,logits_to_keep=len(t)).logits[0].float().log_softmax(-1)
                want=sum(float(lp[i,z]) for i,z in enumerate(t));checks.append(abs(val-want))
        assert max(checks)<1e-3,checks
        dump('SCORER_VERIFIED.json',dict(max_absolute_raw_lp_error=max(checks),n=8,tolerance=1e-3,
             oracle='canonical Qwen3ForCausalLM full causal output, same calibration inputs; answer argmax identity tested separately',errors=checks))
        torch.cuda.reset_peak_memory_stats();forward_rows=0;tokens=0
        with torch.inference_mode(),(a.output/'predictions.jsonl').open('x') as out:
            for i,v in enumerate(rows):
                label_scores=score_labels(model,[r['letter_pairs'] for r in v['rotations']],'cuda:0',pad)
                rotations=[]
                for r,ls in zip(v['rotations'],label_scores):
                    cs,ln=score_pairs(model,r['content_pairs'],'cuda:0',pad)
                    rotations.append(dict(rotation=r['rotation'],order=r['order'],gold_position=r['gold_position'],
                        letter_lp=ls,content_lp=cs,content_tokens=ln,content_mean_lp=[s/n for s,n in zip(cs,ln)]))
                    forward_rows+=5;tokens+=len(r['letter_pairs'][0][0])+sum(len(c)+len(t)-1 for c,t in r['content_pairs'])
                keep={k:v[k] for k in ['id','cluster','task','language','gold','subject','split']}
                out.write(json.dumps({**keep,'rotations':rotations})+'\n')
                if (i+1)%25==0:
                    out.flush();dump('progress.json',dict(completed=i+1,total=len(rows),seconds=time.time()-start,
                         seconds_per_item=(time.time()-start)/(i+1),peak_cuda=torch.cuda.max_memory_allocated()))
        priors=[]
        with torch.inference_mode():
            for v in json.loads((a.data/'content_free.json').read_text()):
                ls=score_labels(model,[v['letter_pairs']],'cuda:0',pad)[0]
                priors.append(dict(task=v['task'],placeholder=v['placeholder'],letter_lp=ls))
        dump('content_free_predictions.json',priors)
        dump('DONE.json',dict(status='COMPLETE',label=a.label,items=len(rows),rotations=len(rows)*4,
             option_scores=len(rows)*4*8,protocol_sha256=sha(a.data/'protocol.json'),prediction_sha256=sha(a.output/'predictions.jsonl'),
             seconds=time.time()-start,gpu_wall_hours=(time.time()-start)/3600,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
             forward_sequences=forward_rows,estimated_input_tokens=tokens,training=False))
    except BaseException as e:
        dump('FAILED.json',dict(error=repr(e),seconds=time.time()-start));raise

if __name__=='__main__':main()
