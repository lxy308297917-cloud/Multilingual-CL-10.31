"""Fixed instruction-domain teacher and projector; no math teacher reuse."""
import os
import time
from common import *


def main():
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM
    assert not CACHE.exists(),'Use a new cache attempt after inspecting failure'
    rows=[json.loads(l) for l in (J/'prepared/alignment.jsonl').open()]
    manifest=read(J/'prepared/manifest.json')
    assert sha(J/'prepared/alignment.jsonl')==manifest['alignment_sha256'] and len(rows)==1024
    CACHE.mkdir(parents=True)
    start=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda').eval()
    for p in model.parameters():p.requires_grad_(False)
    refs={};torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for lang in LANGS:
            values=[]
            for start_row in range(0,len(rows),8):
                batch=[r['encoded'][lang] for r in rows[start_row:start_row+8]]
                length=max(len(r['input_ids']) for r in batch)
                ids=torch.zeros(len(batch),length,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
                for i,r in enumerate(batch):
                    ids[i,:len(r['input_ids'])]=torch.tensor(r['input_ids'],device='cuda');mask[i,:len(r['input_ids'])]=1
                saved=[]
                def hook(module,args,output):saved.append(output[0] if isinstance(output,tuple) else output)
                handle=backbone(model).layers[17].register_forward_hook(hook)
                try:backbone(model)(input_ids=ids,attention_mask=mask,use_cache=False)
                finally:handle.remove()
                values.extend(saved[0][i,r['body_positions'],:].float().mean(0).cpu() for i,r in enumerate(batch))
                if start_row%128==0:print(json.dumps(dict(language=lang,rows=start_row+len(batch))),flush=True)
            refs[lang]=torch.stack(values)
    bases=[];spectra={}
    torch.set_num_threads(4)
    for lang in LANGS[:4]:
        h=refs[lang].double();_,s,vh=torch.linalg.svd(h-h.mean(0),full_matrices=False)
        assert s[127]>1e-8
        bases.append(vh[:128].T);spectra[lang]=s[:128].tolist()
    u,s,_=torch.linalg.svd(torch.cat(bases,dim=1)/2,full_matrices=False)
    shared=u[:,:64].float().contiguous()
    rng=torch.Generator().manual_seed(42)
    random=torch.linalg.qr(torch.randn(2560,64,generator=rng,dtype=torch.float64),mode='reduced')[0].float().contiguous()
    target=sum(refs[l] for l in LANGS[:4])/4
    scales={name:float(((refs['sw'][:64]-target[:64])@basis).square().mean())
            for name,basis in [('shared',shared),('random',random)]}
    assert all(v>1e-8 for v in scales.values())
    assert all(float((x.T@x-torch.eye(64)).abs().max())<1e-5 for x in [random,shared])
    # A real prompt-only activation consistency check, separate from effects.
    checks=[]
    with torch.inference_mode():
        for index in [0,1]:
            r=rows[index]['encoded']['sw'];ids=torch.tensor([r['input_ids']],device='cuda')
            h=pool(model,ids,r['body_positions']).cpu()
            rel=float((h-refs['sw'][index]).norm()/h.norm().clamp_min(1e-8))
            assert rel<.02
            checks.append(dict(id=rows[index]['id'],batch_single_relative_difference=rel))
    save_file(dict(shared=shared,random=random,targets=target.contiguous(),base_sw=refs['sw']),str(CACHE/'teacher.safetensors'))
    dump(CACHE/'metadata.json',dict(ordered_ids=[r['id'] for r in rows],scales=scales,
        fit='Per-reference centered SVD rank128; concatenate bases/2; top64 left singular vectors.',
        readout='1-based block18 raw residual output; mean of instruction+input body tokens, excluding role/format tokens.',
        teacher='Frozen Base per-ID uncentered EN/ZH/FR/DE mean, no reference answers.',
        singular_values=s.tolist(),reference_spectra=spectra,checks=checks,
        elapsed_seconds=time.monotonic()-start,gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    dump(CACHE/'DONE.json',dict(status='BACTRIAN_INSTRUCTION_TEACHER_COMPLETE',
        files={p.name:sha(p) for p in CACHE.iterdir() if p.is_file()},
        input_manifest_sha256=sha(J/'prepared/manifest.json'),source_sha256=sha(__file__)))
    print(json.dumps(dict(status='TEACHER_COMPLETE',scales=scales,seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':main()
