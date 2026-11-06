"""One finite paired development decoding comparison, independent of formal scores."""
from profile_paths import resolve
import os, sys, json, time, argparse, hashlib, datetime, subprocess, collections, re, statistics
from pathlib import Path
R = Path(resolve('ROOT'))
L = Path(resolve('LOGS'))
MODEL = Path(resolve('MODEL'))
ADAPTER = Path(resolve('CHECKPOINTS') + '/lora_final_v1')
if os.environ.get('AYA_EXPECTED_GPU_UUID'):
    sys.path.insert(0, str(R / 'scripts/aya_4b_20261105'))
    import generate_v2_gpu_guard

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--wait-base-generation', action='store_true')
    p.add_argument('--adapter', type=Path, required=True)
    a = p.parse_args()
    ADAPTER = a.adapter
    assert os.environ['CUDA_VISIBLE_DEVICES'] == a.gpu_uuid and os.environ['AYA_EXPECTED_GPU_UUID'] == a.gpu_uuid
    a.output.mkdir(exist_ok=False)
    start = time.monotonic()
    state = a.output / 'state.json'

    def save(status, **kw):
        dump(state, {'status': status, 'pid': os.getpid(), 'updated_at': datetime.datetime.now().astimezone().isoformat(), 'development_only': True, **kw})
    save('WAITING_FORMAL_BASE_GENERATION' if a.wait_base_generation else 'PRECHECK')
    if a.wait_base_generation:
        deadline = time.monotonic() + 21600
        while not (L / 'base-generation-v2/DONE.json').exists():
            if time.monotonic() > deadline:
                save('WAIT_TIMEOUT_NO_RETRY')
                return
            time.sleep(30)
    cfg = json.loads((a.data / 'manifest.json').read_text())
    xs = [json.loads(x) for x in (a.data / 'development.jsonl').open()]
    assert len(xs) == 128 and sha(a.data / 'development.jsonl') == cfg['data_sha256']
    assert json.loads((ADAPTER / 'TRAINING_DONE.json').read_text())['steps'] == 400
    save('WAITING_FREE_GPU')
    deadline = time.monotonic() + 21600
    while True:
        check = subprocess.run([str(Path(resolve('TRAIN_PY'))), str(R / 'scripts/preflight_gpu.py'), '--uuid', a.gpu_uuid, '--max-user-gpus', '4', '--min-free-mib', '21000', '--settle-seconds', '5'], capture_output=True, text=True)
        if check.returncode == 0:
            break
        if time.monotonic() > deadline:
            save('GPU_WAIT_TIMEOUT_NO_RETRY')
            return
        time.sleep(30)
    (a.output / 'gpu-preflight.json').write_text(check.stdout + check.stderr)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prompts = [tok.apply_chat_template(x['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False) for x in xs]
    assert max(map(len, prompts)) <= 7168
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from sacrebleu.metrics import CHRF, BLEU
    manifest = {'development_protocol': cfg, 'data_manifest_sha256': sha(a.data / 'manifest.json'), 'script_sha256': sha(Path(__file__)), 'model_revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'model_weights_sha256': {x.name: sha(x) for x in MODEL.glob('*.safetensors')}, 'adapter_weights_sha256': sha(ADAPTER / 'adapter_model.safetensors'), 'prompt_tokens_sha256': hashlib.sha256(json.dumps(prompts, separators=(',', ':')).encode()).hexdigest(), 'backend': {'max_num_seqs': 8, 'max_num_batched_tokens': 4096, 'gpu_memory_utilization': 0.65, 'enable_lora': True, 'batch_size': 8, 'dtype': 'bfloat16', 'max_model_len': 8192, 'tensor_parallel_size': 1, 'seed': 42}, 'started_at': datetime.datetime.now().astimezone().isoformat()}
    dump(a.output / 'manifest.json', manifest)
    save('LOADING_ONE_SHARED_ENGINE')
    generation_start = time.monotonic()
    engine = LLM(model=str(MODEL), tokenizer=str(MODEL), dtype='bfloat16', max_model_len=8192, max_num_seqs=8, max_num_batched_tokens=4096, gpu_memory_utilization=0.65, seed=42, enable_lora=True, max_lora_rank=16, generation_config='vllm', trust_remote_code=False)
    adapter = LoRARequest('fixed-final400', 1, str(ADAPTER))
    summary = []
    tokens = 0
    with (a.output / 'predictions.jsonl').open('x') as stream:
        for method, request in [('base', None), ('lora', adapter)]:
            for profile, settings in cfg['profiles'].items():
                save('GENERATING_DEVELOPMENT', method=method, profile=profile)
                params = SamplingParams(n=1, **settings, max_tokens=cfg['max_new_tokens'], stop_token_ids=cfg['stop_token_ids'], seed=cfg['seed'], skip_special_tokens=True)
                grouped = collections.defaultdict(list)
                for offset in range(0, len(xs), 8):
                    batch = xs[offset:offset + 8]
                    output = engine.generate([{'prompt_token_ids': x} for x in prompts[offset:offset + 8]], params, lora_request=request, use_tqdm=False)
                    assert len(output) == len(batch)
                    for row, result in zip(batch, output):
                        y = result.outputs[0]
                        r = {k: row[k] for k in ['id', 'pair_index', 'direction', 'source', 'reference']}
                        r.update(model=method, profile=profile, prediction=y.text.strip(), generated_token_ids=list(y.token_ids), generated_tokens=len(y.token_ids), finish_reason=y.finish_reason)
                        stream.write(json.dumps(r, ensure_ascii=False) + '\n')
                        grouped[row['direction']].append(r)
                        tokens += len(y.token_ids)
                    stream.flush()
                    print(json.dumps({'model': method, 'profile': profile, 'completed': min(offset + 8, len(xs)), 'total': len(xs)}), flush=True)
                for direction, items in grouped.items():
                    pred = [x['prediction'] for x in items]
                    ref = [[x['reference'] for x in items]]
                    chrf = CHRF(word_order=2).corpus_score(pred, ref)
                    bleu = BLEU(tokenize='13a').corpus_score(pred, ref)
                    repeated = 0
                    for x in items:
                        words = re.findall('\\w+', x['prediction'].casefold())
                        grams = collections.Counter((tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))))
                        repeated += len(words) >= 24 and max(grams.values(), default=0) >= 5 and (len(grams) / max(1, sum(grams.values())) < 0.5)
                    summary.append({'model': method, 'profile': profile, 'direction': direction, 'samples': len(items), 'chrF++': chrf.score, 'BLEU': bleu.score, 'cap_count': sum((x['finish_reason'] == 'length' for x in items)), 'repetitive_4gram_count': repeated, 'empty_count': sum((not x['prediction'] for x in items)), 'mean_generated_tokens': statistics.mean((x['generated_tokens'] for x in items))})
                    print(json.dumps(summary[-1]), flush=True)
    dump(a.output / 'summary.json', {'status': 'DEVELOPMENT_COMPLETE_NOT_FORMAL_TEST', 'rows': summary, 'data_sha256': cfg['data_sha256'], 'predictions_sha256': sha(a.output / 'predictions.jsonl'), 'inference_seed_count': 1, 'no_test_or_checkpoint_selection': True})
    dump(a.output / 'cost.json', {'generation_process_seconds': time.monotonic() - generation_start, 'including_wait_process_seconds': time.monotonic() - start, 'generated_tokens': tokens, 'completed_generations': 512, 'scope': 'Additional development decoding diagnosis, excluded from original formal evaluation cost table'})
    save('COMPLETE')
    dump(a.output / 'DONE.json', {'status': 'DEVELOPMENT_COMPLETE', 'generations': 512, 'model_checkpoints': ['base', 'lora_final400']})
if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
