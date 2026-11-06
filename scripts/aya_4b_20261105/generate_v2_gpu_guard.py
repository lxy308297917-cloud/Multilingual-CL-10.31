"""vLLM thin runner for MT, summary and code; generation never executes code."""
from profile_paths import resolve
import argparse, json, os, sys, time, subprocess, importlib.metadata as metadata
sys.path.insert(0, resolve('ROOT') + '/scripts/bench_v2')
from pathlib import Path
from common import sha256, dump, object_hash, checked_rows, ids_hash, PredictionStore
from prompting import render
if os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/sft_v1'))
    from vllm_compat import install as _install_uuid_mapping
    _install_uuid_mapping()
if os.environ.get('AYA_EXPECTED_GPU_UUID'):
    import torch
    from vllm.v1.worker.gpu_worker import Worker as _GPUWorker
    _original_init_device = _GPUWorker.init_device

    def _guarded_init_device(self):
        expected = os.environ['AYA_EXPECTED_GPU_UUID'].removeprefix('GPU-')
        actual = str(torch.cuda.get_device_properties(self.local_rank).uuid)
        print(json.dumps({'gpu_identity_guard': 'worker_before_init', 'pid': os.getpid(), 'visible': os.environ.get('CUDA_VISIBLE_DEVICES'), 'expected': expected, 'actual': actual, 'memory': torch.cuda.mem_get_info(self.local_rank)}), flush=True)
        if actual != expected:
            raise RuntimeError('GPU physical identity differs; refusing model allocation')
        return _original_init_device(self)
    _GPUWorker.init_device = _guarded_init_device

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--method', choices=['base', 'full', 'lora', 'ssu'], required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scope', choices=['smoke', 'pilot', 'full'], required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-user-gpus', type=int, default=4)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu_uuid:
        raise ValueError('Explicit single GPU UUID must match CUDA_VISIBLE_DEVICES')
    if (a.method == 'lora') != (a.adapter is not None):
        raise ValueError('LoRA must use adapter; other methods must not')
    if a.batch_size < 1:
        raise ValueError('Invalid batch size')
    cfg = json.loads((a.dataset / 'manifest.json').read_text())
    data = a.dataset / 'test.jsonl'
    rows = checked_rows(data, cfg['samples'])
    if sha256(data) != cfg['data_sha256'] or ids_hash(rows) != cfg['sample_ids_sha256']:
        raise ValueError('Dataset identity differs')
    expected = [r['id'] for r in rows] if a.scope == 'full' else cfg[a.scope + '_ids']
    selected = {r['id']: r for r in rows}
    chosen = [selected[id] for id in expected]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    prompts = {r['id']: render(tok, r, cfg['max_input_tokens']) for r in chosen}
    paths = [a.model / 'config.json', a.tokenizer / 'tokenizer.json', a.tokenizer / 'tokenizer_config.json']
    paths += sorted(a.model.glob('*.safetensors'))
    if a.adapter:
        paths += [a.adapter / 'adapter_config.json', a.adapter / 'adapter_model.safetensors']
    model_identity = {str(path.resolve()): sha256(path) for path in paths}
    code = {name: sha256(Path(__file__) if name == 'generate.py' else Path(resolve('ROOT') + '/scripts/bench_v2') / name) for name in ['common.py', 'prompting.py', 'generate.py']}
    manifest = dict(result_schema_version='bench-v2.1', execution_profile='aya4b-cuda-memory-v2', benchmark_id=cfg['benchmark_id'], benchmark_protocol=cfg, method=a.method, training_seed=None if a.method == 'base' else 42, model_revision='1cfa9a7208912126459214e8b04321603b3df60c', checkpoint_identity=model_identity, code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True).strip(), source_files_sha256=code, data_manifest_hash=sha256(a.dataset / 'manifest.json'), sample_ids=expected, sample_ids_sha256=object_hash(expected), scope=a.scope, tokenizer=str(a.tokenizer.resolve()), prompt_tokens_sha256=object_hash({id: prompts[id]['prompt_token_ids'] for id in expected}), backend={'vllm': metadata.version('vllm'), 'transformers': metadata.version('transformers'), 'torch': metadata.version('torch'), 'dtype': 'bfloat16', 'max_model_len': cfg['max_model_len'], 'max_num_seqs': 8, 'max_num_batched_tokens': 4096, 'gpu_memory_utilization': 0.65, 'tensor_parallel_size': 1, 'generation_config': 'vllm'}, batch_size=a.batch_size)
    store = PredictionStore(a.output, manifest, expected, resume=a.resume)
    todo = store.pending()
    if not todo:
        print(json.dumps(store.finish(a.scope)))
        return
    root = Path(__file__).resolve().parents[2]
    preflight = subprocess.run([sys.executable, str(root / 'scripts/preflight_gpu.py'), '--uuid', a.gpu_uuid, '--max-user-gpus', str(a.max_user_gpus), '--min-free-mib', '21000'], capture_output=True, text=True)
    (a.output / 'gpu-preflight.json').write_text(preflight.stdout + preflight.stderr)
    if preflight.returncode:
        raise RuntimeError('GPU preflight rejected resource; no CUDA model loaded')
    sys.path.insert(0, str(root / 'scripts/sft_v1'))
    from vllm_compat import install
    install()
    from vllm import LLM, SamplingParams
    start = time.monotonic()
    engine = LLM(model=str(a.model), tokenizer=str(a.tokenizer), dtype='bfloat16', tensor_parallel_size=1, max_model_len=cfg['max_model_len'], max_num_seqs=8, max_num_batched_tokens=4096, gpu_memory_utilization=0.65, seed=42, enable_lora=a.adapter is not None, max_lora_rank=16, generation_config='vllm', disable_log_stats=False, trust_remote_code=False)
    adapter = None
    if a.adapter:
        from vllm.lora.request import LoRARequest
        adapter = LoRARequest('aya-sw6400-final', 1, str(a.adapter))
    d = cfg['decoding_config']
    sampling = SamplingParams(n=1, temperature=d['temperature'], top_p=d['top_p'], top_k=d['top_k'], max_tokens=d['max_new_tokens'], stop_token_ids=d['stop_token_ids'], seed=d['seed'], skip_special_tokens=True)
    for offset in range(0, len(todo), a.batch_size):
        batch = todo[offset:offset + a.batch_size]
        outputs = engine.generate([{'prompt_token_ids': prompts[id]['prompt_token_ids']} for id in batch], sampling, lora_request=adapter, use_tqdm=False)
        out = []
        if len(outputs) != len(batch):
            raise ValueError('vLLM returned missing requests')
        for id, request in zip(batch, outputs):
            if len(request.outputs) != 1 or list(request.prompt_token_ids) != prompts[id]['prompt_token_ids']:
                raise ValueError('Candidate count or prompt drift')
            completion = request.outputs[0]
            row = selected[id]
            r = {k: v for k, v in row.items() if k not in ['messages', 'article_prompt_template']}
            r.update({k: v for k, v in prompts[id].items() if k != 'prompt_token_ids'})
            r.update(raw_output=completion.text, prediction=completion.text.strip(), generated_token_ids=list(completion.token_ids), generated_tokens=len(completion.token_ids), finish_reason=completion.finish_reason, stop_reason=completion.stop_reason, prompt_token_sha256=object_hash(prompts[id]['prompt_token_ids']), generation_hit_cap=completion.finish_reason == 'length', method=a.method)
            out.append(r)
        store.append(out)
        print(json.dumps(dict(completed=len(store.rows), total=len(expected), scope=a.scope, elapsed_seconds=time.monotonic() - start)), flush=True)
    import torch
    cost = dict(elapsed_seconds=time.monotonic() - start, input_tokens=sum((r['input_tokens'] for r in store.rows.values())), generated_tokens=sum((r['generated_tokens'] for r in store.rows.values())), gpu_uuid=a.gpu_uuid, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), torch_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), note='allocated/reserved do not include all vLLM GPU allocations; wall time excludes CPU prompt/hash preparation')
    dump(a.output / 'cost.json', cost)
    integrity = store.finish(a.scope)
    dump(a.output / 'DONE.json', integrity)
    print(json.dumps(integrity))
if __name__ == '__main__':
    main()
