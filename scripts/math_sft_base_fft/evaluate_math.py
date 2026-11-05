"""Immutable greedy generation; PolyMath equivalence scored separately on CPU."""
import argparse, os, sys, time, collections, json
from pathlib import Path
from common import *
if os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
    sys.path.insert(0, str(ROOT / 'scripts/sft_v1'))
    from vllm_compat import install
    install()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--langs', nargs='+', required=True)
    ap.add_argument('--model', type=Path, default=MODEL)
    ap.add_argument('--role', default='base')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert gpu.startswith('GPU-') and ',' not in gpu
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    d = DATA / 'prepared-v1'
    protocol = json.loads((d / 'protocol.json').read_text())
    rs = []
    assets = {}
    for lang in a.langs:
        p = d / f'{lang}_eval.jsonl'
        assert sha(p) == protocol['files'][p.name]
        rs += rows(p)
        assets[p.name] = sha(p)
    weights = {p.name: sha(p) for p in sorted(a.model.glob('*.safetensors'))}
    assert weights
    if a.model == MODEL:
        assert weights == protocol['weights_sha256']
    from vllm import LLM, SamplingParams
    start = time.monotonic()
    gen = protocol['generation']
    engine = LLM(model=str(a.model), tokenizer=str(MODEL), dtype='bfloat16', tensor_parallel_size=1, max_model_len=gen['max_model_len'], max_num_seqs=gen['max_num_seqs'], max_num_batched_tokens=gen['max_num_batched_tokens'], gpu_memory_utilization=gen['gpu_memory_utilization'], seed=42, generation_config='vllm', enable_prefix_caching=True, trust_remote_code=False)
    dump(a.output / 'run.json', dict(role=a.role, model=str(a.model), weights_sha256=weights, protocol_sha256=sha(d / 'protocol.json'), assets=assets, generation=gen, gpu_uuid=gpu, code_sha256=sha(__file__), common_sha256=sha(Path(__file__).with_name('common.py'))))
    preds = []
    with (a.output / 'predictions.jsonl').open('x') as f:
        for off in range(0, len(rs), 128):
            batch = rs[off:off + 128]
            params = [SamplingParams(temperature=0, top_p=1, max_tokens=2048 if r['benchmark'] == 'polymath' else 1280, seed=42) for r in batch]
            outs = engine.generate([{'prompt_token_ids': r['prompt_token_ids']} for r in batch], params, use_tqdm=False)
            assert len(outs) == len(batch)
            for r, o in zip(batch, outs):
                assert list(o.prompt_token_ids) == r['prompt_token_ids']
                c = o.outputs[0]
                x = {k: v for k, v in r.items() if k != 'prompt_token_ids'}
                x.update(output=c.text, generated_tokens=len(c.token_ids), finish_reason=c.finish_reason)
                if r['benchmark'] != 'polymath':
                    x['parsed'] = number(c.text, r['lang'])
                    x['strict_parsed'] = strict_number(c.text)
                    x['correct'] = x['parsed'] is not None and x['parsed'] == x['gold_numeric']
                    x['strict_correct'] = x['strict_parsed'] is not None and x['strict_parsed'] == x['gold_numeric']
                f.write(json.dumps(x, ensure_ascii=False) + '\n')
                preds.append(x)
            f.flush()
            print(json.dumps(dict(done=len(preds), total=len(rs), seconds=time.monotonic() - start)), flush=True)
    groups = collections.defaultdict(list)
    for x in preds:
        groups[x['lang'] + ':' + x['benchmark'] + (':' + x['level'] if 'level' in x else '')].append(x)
    scores = {}
    for k, xs in groups.items():
        scores[k] = {'samples': len(xs), 'hit_cap_rate': sum((x['finish_reason'] == 'length' for x in xs)) / len(xs), 'generated_tokens': sum((x['generated_tokens'] for x in xs)), 'scoring_status': 'PENDING_OFFICIAL_EQUIVALENCE' if xs[0]['benchmark'] == 'polymath' else 'COMPLETE'}
        if xs[0]['benchmark'] != 'polymath':
            scores[k].update(correct=sum((x['correct'] for x in xs)), accuracy=sum((x['correct'] for x in xs)) / len(xs), strict_accuracy=sum((x['strict_correct'] for x in xs)) / len(xs), parse_rate=sum((x['parsed'] is not None for x in xs)) / len(xs))
    dump(a.output / 'scores.json', scores)
    dump(a.output / 'GENERATION_DONE.json', {'status': 'COMPLETE', 'samples': len(preds), 'seconds': time.monotonic() - start, 'input_tokens': sum((len(r['prompt_token_ids']) for r in rs)), 'generated_tokens': sum((r['generated_tokens'] for r in preds)), 'predictions_sha256': sha(a.output / 'predictions.jsonl')})
    print(json.dumps(scores), flush=True)
if __name__ == '__main__':
    main()
