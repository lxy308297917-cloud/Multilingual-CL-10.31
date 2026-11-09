"""Frozen block features only; no logits, optimizer, backward or generated answers."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/data0/users/lixiaoyang/10.31/multilingual-research')
D=Path('/data1/lixiaoyang/datasets/representation-benchmarks-20261109/v1')
OLD=Path('/data1/lixiaoyang/datasets/functional-representation-20261109/p1-v1')
P='/data1/lixiaoyang/envs/qwen3-multilingual-20261031/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def worker(a):
    assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['AYA_EXPECTED_GPU_UUID']==a.gpu_uuid
    a.cache.mkdir(parents=True,exist_ok=False)
    pre=subprocess.run([P,str(R/'scripts/preflight_gpu.py'),'--uuid',a.gpu_uuid,'--max-user-gpus','4','--min-free-mib','21000'],capture_output=True,text=True)
    (a.cache/'preflight.json').write_text(pre.stdout+pre.stderr);assert pre.returncode==0
    import torch
    from transformers import AutoModelForCausalLM
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    assert str(torch.cuda.get_device_properties(0).uuid)==a.gpu_uuid.removeprefix('GPU-')
    assets=json.loads((R/'configs/representation_benchmarks_20261109/assets.json').read_text());m=assets['models'][a.model]
    if m['adapter']:assert sha(Path(m['adapter'])/'adapter_model.safetensors')==m['adapter_sha256']
    model=AutoModelForCausalLM.from_pretrained(m['base_path'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda')
    if m['adapter']:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,m['adapter'],is_trainable=False)
    model.eval();model.requires_grad_(False);raw=model.get_base_model() if hasattr(model,'get_base_model') else model
    assert len(raw.model.layers)==36 and model.config.hidden_size==2560
    def forward(ids,positions=None,masks=None):
        ids=torch.tensor([ids],device='cuda',dtype=torch.long);values={};handles=[]
        def save(layer,output):
            h=(output[0] if isinstance(output,tuple) else output)[0]
            if positions is not None:values[layer]=h[positions].detach().cpu()
            else:values[layer]=torch.stack([h[index].float().mean(0) for index in masks]).detach().cpu()
        if positions is None:handles.append(raw.model.embed_tokens.register_forward_hook(lambda mod,inp,out:save(0,out)))
        for i,b in enumerate(raw.model.layers):handles.append(b.register_forward_hook(lambda mod,inp,out,j=i+1:save(j,out)))
        try:
            with torch.inference_mode():raw.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
        finally:
            for h in handles:h.remove()
        assert len(values)==(36 if positions is not None else 37)
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        v=torch.stack([values[i] for i in sorted(values)],dim=1)
        assert torch.isfinite(v).all();return v
    start=time.monotonic();old=list(map(json.loads,(OLD/'encoded_selected.jsonl').open()));assert len(old)==256
    assert sha(OLD/'encoded_selected.jsonl')==assets['old_pool']['encoded_sha256']
    if a.engineering:old=old[:2]
    features=[];meta=[];offset=0
    for i,row in enumerate(old):
        f=forward(row['input_ids'],positions=row['positions']);features.append(f)
        meta.append({k:v for k,v in row.items() if k!='input_ids'}|dict(feature_offset_start=offset,feature_offset_stop=offset+len(f)));offset+=len(f)
        if (i+1)%32==0:print(json.dumps(dict(stage='old',completed=i+1,total=256,elapsed=time.monotonic()-start)),flush=True)
    save_file({'features':torch.cat(features).contiguous()},str(a.cache/'old.safetensors'));(a.cache/'old_rows.json').write_text(json.dumps(meta,indent=2)+'\n');del features
    manifest=json.loads((D/'parallel/manifest.json').read_text())
    assert sha(D/'parallel/encoded.jsonl')==manifest['files']['encoded.jsonl']
    pairs=list(map(json.loads,(D/'parallel/encoded.jsonl').open()));assert len(pairs)==300
    if a.engineering:pairs=pairs[:2]
    pooled=[]
    for i,row in enumerate(pairs):
        langs=[]
        for lang in ['en','sw']:
            x=row['languages'][lang];raw_mean=forward(x['raw_ids'],masks=[list(range(len(x['raw_ids'])))])
            chat_means=forward(x['chat_ids'],masks=[list(range(len(x['chat_ids']))),x['chat_content_positions']])
            langs.append(torch.cat([raw_mean,chat_means]))
        pooled.append(torch.stack(langs))
        if (i+1)%25==0:print(json.dumps(dict(stage='parallel',completed=i+1,total=300,elapsed=time.monotonic()-start)),flush=True)
    t=torch.stack(pooled);assert t.shape==(len(pairs),2,3,37,2560)
    save_file({'pooled':t.contiguous()},str(a.cache/'parallel.safetensors'))
    cost=dict(model=a.model,status='ENGINEERING_ONLY' if a.engineering else 'COMPLETE',old_rows=len(old),pairs=len(pairs),conditions=['raw_content_mean','chat_all_tokens_mean','chat_content_tokens_mean'],layer_convention='0 embedding;1..36 raw block output BEFORE final norm; no padded/silent truncation tokens',seconds=time.monotonic()-start,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),model_identity=m,all_parameters_frozen=True,new_training_jobs=0,hashes={n:sha(a.cache/n) for n in ['old.safetensors','old_rows.json','parallel.safetensors']},parallel_manifest_sha256=sha(D/'parallel/manifest.json'))
    if m['adapter']:assert sha(Path(m['adapter'])/'adapter_model.safetensors')==m['adapter_sha256']
    (a.cache/'DONE.json').write_text(json.dumps(cost,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=['Base','all_layer','full_feature','ce_replay']);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--worker',action='store_true');ap.add_argument('--gpu-uuid');ap.add_argument('--engineering',action='store_true');a=ap.parse_args()
    if a.worker:return worker(a)
    a.output.mkdir(parents=True,exist_ok=False)
    def save(status,**more):(a.output/'state.json').write_text(json.dumps(dict(status=status,model=a.model,pid=os.getpid(),updated_at=datetime.datetime.now().astimezone().isoformat(),**more),indent=2)+'\n')
    sys.path.insert(0,str(R/'scripts/aya_4b_multilang_20261106'));from gpu_slot import run_job
    env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4'}
    try:
        run_job(lambda uuid:[P,__file__,'--worker','--model',a.model,'--cache',a.cache,'--output',a.output,'--gpu-uuid',uuid,*(['--engineering'] if a.engineering else [])],env,a.output/'extract.log',save)
        assert (a.cache/'DONE.json').exists();save('COMPLETE')
    except Exception as e:save('FAILED',error=repr(e));raise
if __name__=='__main__':main()
