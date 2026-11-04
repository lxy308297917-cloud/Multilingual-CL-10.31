"""Frozen generative task evaluation using existing vLLM, with raw outputs.

Classification generation is primary. Teacher-forced label log likelihood on a
frozen dev64 subset is a diagnostic, never substituted for generation scores.
"""
from runtime import runtime_path
import argparse, collections, json, os, re, sys, time
from pathlib import Path
from common import ROOT, DATA, MODEL, LANGS, sha, oh, dump, rows, number, numeric_prediction, order
if os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
    sys.path.insert(0, str(ROOT / 'scripts/sft_v1'))
    from vllm_compat import install
    install()

def tree(s):
    toks = re.findall('\\[|\\]|[^\\s\\[\\]]+', s)
    stack = []
    roots = []
    for t in toks:
        if t == '[':
            stack.append([])
        elif t == ']':
            if not stack:
                raise ValueError('Unmatched close')
            node = stack.pop()
            if not node or not isinstance(node[0], str) or (not re.fullmatch('(IN|SL):[A-Z_]+', node[0])):
                raise ValueError('Invalid node')
            if stack:
                stack[-1].append(node)
            else:
                roots.append(node)
        else:
            if not stack:
                raise ValueError('Text outside tree')
            stack[-1].append(t)
    if stack or len(roots) != 1 or (not roots[0][0].startswith('IN:')):
        raise ValueError('Invalid root')
    return roots[0]

def metrics(rs):
    task = rs[0]['track']
    correct = []
    extra = {}
    if task in ['massive', 'xnli', 'sib200']:
        for r in rs:
            pred = r['output'].strip()
            r['parsed'] = pred
            r['correct'] = pred == r['answer']
            r['valid'] = pred in r['labels']
            correct.append(r['correct'])
        cs = collections.defaultdict(list)
        for r in rs:
            cs[r['answer']].append(r['correct'])
        macro_acc = sum((sum(v) / len(v) for v in cs.values())) / len(cs)
        f1 = []
        for label in sorted({r['answer'] for r in rs}):
            tp = sum((r['parsed'] == label and r['answer'] == label for r in rs))
            fp = sum((r['parsed'] == label and r['answer'] != label for r in rs))
            fn = sum((r['parsed'] != label and r['answer'] == label for r in rs))
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
        extra.update(invalid_label_rate=sum((not r['valid'] for r in rs)) / len(rs), macro_accuracy=macro_acc, macro_f1=sum(f1) / len(f1), prediction_counts=dict(collections.Counter((r['parsed'] for r in rs))))
    elif task == 'mathoctopus':
        for r in rs:
            r['parsed'] = number(r['output'], r['lang'])
            r['strict_parsed'] = numeric_prediction(r['output'])
            r['correct'] = r['parsed'] is not None and r['parsed'] == r['gold_numeric']
            r['strict_correct'] = r['strict_parsed'] is not None and r['strict_parsed'] == r['gold_numeric']
            correct.append(r['correct'])
        extra.update(strict_accuracy=sum((r['strict_correct'] for r in rs)) / len(rs), answer_parse_rate=sum((r['parsed'] is not None for r in rs)) / len(rs))
    elif task == 'mtop':
        for r in rs:
            gold = tree(r['answer'])
            try:
                pred = tree(r['output'].strip())
                r['valid'] = True
            except ValueError:
                pred = None
                r['valid'] = False
            r['correct'] = pred == gold
            r['intent_correct'] = pred is not None and pred[0] == gold[0]
            correct.append(r['correct'])
        extra.update(intent_accuracy=sum((r['intent_correct'] for r in rs)) / len(rs), valid_tree_rate=sum((r['valid'] for r in rs)) / len(rs))
    else:
        raise ValueError('Summary protocol must be audited before enabling ROUGE')
    return {'samples': len(rs), 'primary_metric': 'structural_exact_match' if task == 'mtop' else 'accuracy', 'score': sum(correct) / len(correct), 'hit_cap_rate': sum((r['finish_reason'] == 'length' for r in rs)) / len(rs), **extra}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', required=True)
    ap.add_argument('--model', type=Path, default=MODEL)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--langs', nargs='+', default=LANGS)
    ap.add_argument('--max-num-seqs', type=int, default=32)
    ap.add_argument('--candidate-dev', action='store_true')
    a = ap.parse_args()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert gpu.startswith('GPU-') and ',' not in gpu
    assert not a.output.exists(), a.output
    a.output.mkdir(parents=True)
    version = os.environ.get('UNIFIED_DATA_VERSION', 'prepared-v1')
    d = DATA / version / a.track
    protocol = json.loads((d / 'protocol.json').read_text())
    rs = []
    assets = {}
    for lang in a.langs:
        p = d / f'{lang}_{a.split}.jsonl'
        assert sha(p) == protocol['files'][p.name]
        rs += rows(p)
        assets[p.name] = sha(p)
    assert rs
    weight_files = sorted(a.model.glob('*.safetensors'))
    assert weight_files
    weights = {p.name: sha(p) for p in weight_files}
    from vllm import LLM, SamplingParams
    max_new = 1280 if a.track == 'mathoctopus' else 256 if a.track == 'mtop' else 24
    start = time.monotonic()
    engine = LLM(model=str(a.model), tokenizer=str(MODEL), dtype='bfloat16', tensor_parallel_size=1, max_model_len=8192, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=8192, gpu_memory_utilization=0.8, seed=42, generation_config='vllm', enable_prefix_caching=True, trust_remote_code=False)
    sampling = SamplingParams(temperature=0, top_p=1, max_tokens=max_new, seed=42)
    dump(a.output / 'run.json', dict(model=str(a.model), weights_sha256=weights, protocol_sha256=sha(d / 'protocol.json'), assets=assets, task=a.track, split=a.split, langs=a.langs, max_new_tokens=max_new, decoding='greedy', max_model_len=8192, source_code_sha256=sha(__file__), prompt_source='Exact upstream LF qwen3 enable_thinking=False source IDs; no few-shot examples', gpu_uuid=gpu))
    preds = []
    with (a.output / 'predictions.jsonl').open('x') as stream:
        for off in range(0, len(rs), 128):
            batch = rs[off:off + 128]
            outs = engine.generate([{'prompt_token_ids': r['prompt_token_ids']} for r in batch], sampling, use_tqdm=False)
            assert len(outs) == len(batch)
            for r, out in zip(batch, outs):
                assert list(out.prompt_token_ids) == r['prompt_token_ids']
                c = out.outputs[0]
                item = {k: v for k, v in r.items() if k not in ['prompt_token_ids']}
                item.update(output=c.text, generated_tokens=len(c.token_ids), finish_reason=c.finish_reason, prompt_token_sha256=oh(r['prompt_token_ids']))
                preds.append(item)
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            stream.flush()
            print(json.dumps(dict(track=a.track, split=a.split, done=len(preds), total=len(rs), seconds=time.monotonic() - start)), flush=True)
    groups = collections.defaultdict(list)
    for r in preds:
        groups[r['lang'] + ':' + r.get('benchmark', a.track)].append(r)
    scores = {k: metrics(v) for k, v in groups.items()}
    if a.candidate_dev:
        assert a.split == 'dev' and a.track in ['massive', 'xnli', 'sib200']
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        candidate_rows = []
        sample = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1, seed=42)
        for lang in a.langs:
            subset = sorted([r for r in rs if r['lang'] == lang], key=lambda r: oh([42, r['id']]))[:64]
            for r in subset:
                prompts = []
                starts = []
                for label in protocol['labels']:
                    target = tok.encode(label, add_special_tokens=False)
                    prompts.append({'prompt_token_ids': r['prompt_token_ids'] + target})
                    starts.append(len(r['prompt_token_ids']))
                vals = []
                for off in range(0, len(prompts), 16):
                    outs = engine.generate(prompts[off:off + 16], sample, use_tqdm=False)
                    for out, begin in zip(outs, starts[off:off + 16]):
                        lp = out.prompt_logprobs
                        ids = out.prompt_token_ids
                        assert lp is not None and len(lp) == len(ids)
                        vals.append(sum((float(lp[j][ids[j]].logprob) for j in range(begin, len(ids)))))
                pred = protocol['labels'][max(range(len(vals)), key=lambda j: vals[j])]
                candidate_rows.append(dict(id=r['id'], lang=lang, scores=vals, prediction=pred, answer=r['answer'], correct=pred == r['answer']))
            print(json.dumps({'candidate_lang_done': lang, 'rows': len(subset)}), flush=True)
        (a.output / 'candidate_predictions.jsonl').write_text(''.join((json.dumps(x) + '\n' for x in candidate_rows)))
        scores['candidate_dev64'] = {l: {'samples': sum((x['lang'] == l for x in candidate_rows)), 'accuracy': sum((x['correct'] for x in candidate_rows if x['lang'] == l)) / sum((x['lang'] == l for x in candidate_rows))} for l in a.langs}
    dump(a.output / 'scores.json', scores)
    dump(a.output / 'DONE.json', dict(status='DONE', samples=len(preds), seconds=time.monotonic() - start, input_tokens=sum((len(r['prompt_token_ids']) for r in rs)), generated_tokens=sum((r['generated_tokens'] for r in preds)), predictions_sha256=sha(a.output / 'predictions.jsonl'), scores_sha256=sha(a.output / 'scores.json')))
    print(json.dumps(scores, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
