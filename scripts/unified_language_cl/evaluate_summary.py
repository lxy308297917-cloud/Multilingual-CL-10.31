"""Greedy native XL-Sum evaluation; exact author ROUGE and auxiliary language."""
from runtime import runtime_path
import argparse, collections, json, os, sys, time
from pathlib import Path
from common import ROOT, DATA, MODEL, LANGS, sha, oh, dump, rows
from summary_scoring import score_rows
if os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
    sys.path.insert(0, str(ROOT / 'scripts/sft_v1'))
    from vllm_compat import install
    install()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', choices=['xlsum', 'wikilingua'], default='xlsum')
    ap.add_argument('--model', type=Path, default=MODEL)
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--langs', nargs='+', default=LANGS)
    a = ap.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    d = DATA / 'prepared-v1' / a.track
    protocol = json.loads((d / 'protocol.json').read_text())
    rs = []
    assets = {}
    for lang in a.langs:
        p = d / f'{lang}_{a.split}.jsonl'
        assert sha(p) == protocol['files'][p.name]
        assets[p.name] = sha(p)
        rs += rows(p)
    weights = {p.name: sha(p) for p in a.model.glob('*.safetensors')}
    assert weights
    from vllm import LLM, SamplingParams
    from langdetect import DetectorFactory, detect_langs, LangDetectException
    DetectorFactory.seed = 42
    start = time.monotonic()
    engine = LLM(model=str(a.model), tokenizer=str(MODEL), dtype='bfloat16', tensor_parallel_size=1, max_model_len=4096, max_num_seqs=32, max_num_batched_tokens=8192, gpu_memory_utilization=0.8, seed=42, generation_config='vllm', enable_prefix_caching=True, trust_remote_code=False)
    sampling = SamplingParams(temperature=0, top_p=1, max_tokens=512, seed=42)
    dump(a.output / 'run.json', dict(model=str(a.model), weights_sha256=weights, protocol_sha256=sha(d / 'protocol.json'), assets=assets, task=a.track, split=a.split, langs=a.langs, max_new_tokens=512, decoding='greedy', max_model_len=4096, source_code_sha256=sha(__file__), gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], scorer_sha256=sha(Path(__file__).with_name('summary_scoring.py')), primary_scorer_version='v2_explicit_multilingual_sentence_tokenizer', score_correction_protocol_sha256=sha(Path(runtime_path('configs', '')) / 'SUMMARY_SCORING_CORRECTION_v2.json')))
    preds = []
    with (a.output / 'predictions.jsonl').open('x') as stream:
        for off in range(0, len(rs), 128):
            batch = rs[off:off + 128]
            outs = engine.generate([{'prompt_token_ids': r['prompt_token_ids']} for r in batch], sampling, use_tqdm=False)
            assert len(outs) == len(batch)
            for r, out in zip(batch, outs):
                assert out.prompt_token_ids == r['prompt_token_ids']
                c = out.outputs[0]
                pred = c.text.strip()
                detected = None
                confidence = None
                if sum((ch.isalpha() for ch in pred)) >= 20:
                    try:
                        guess = detect_langs(pred[:8000])[0]
                        confidence = guess.prob
                        if confidence >= 0.8:
                            detected = guess.lang
                    except LangDetectException:
                        pass
                item = {k: v for k, v in r.items() if k not in ['prompt_token_ids', 'training_answer']}
                item.update(output=c.text, prediction=pred, generated_tokens=len(c.token_ids), finish_reason=c.finish_reason, prompt_token_sha256=oh(r['prompt_token_ids']), aux_detected_language=detected, aux_language_confidence=confidence)
                preds.append(item)
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            stream.flush()
            print(json.dumps(dict(task=a.track, split=a.split, done=len(preds), total=len(rs), seconds=time.monotonic() - start)), flush=True)
    assert len(preds) == len(rs)
    scores = {}
    scored = []
    for lang in a.langs:
        group = [r for r in preds if r['lang'] == lang]
        vals = score_rows([r['prediction'] for r in group], [r['answer'] for r in group], lang)
        for r, s in zip(group, vals):
            scored.append({'id': r['id'], 'lang': lang, **s})
        n = len(group)
        known = [r for r in group if r['aux_detected_language'] is not None]
        scores[lang + ':' + a.track] = {'samples': n, 'primary_metric': 'rougeLsum', 'score': sum((x['rougeLsum'] for x in vals)) / n, **{k: sum((x[k] for x in vals)) / n for k in vals[0]}, 'mean_generated_tokens': sum((r['generated_tokens'] for r in group)) / n, 'mean_reference_tokens': sum((r['assistant_tokens_untruncated'] for r in group)) / n, 'hit_cap_rate': sum((r['finish_reason'] == 'length' for r in group)) / n, 'source_truncated_rate': sum((r['source_truncated'] for r in group)) / n, 'language_aux_confident_n': len(known), 'language_aux_correct_n': sum((r['aux_detected_language'] == lang for r in known)), 'language_aux_limit': 'langdetect1.0.9 automatic diagnostic only, not human language correctness ground truth; short/low-confidence outputs are unmeasured.'}
    (a.output / 'per_sample_scores.jsonl').write_text(''.join((json.dumps(x) + '\n' for x in scored)))
    dump(a.output / 'scores.json', scores)
    dump(a.output / 'DONE.json', dict(status='DONE', samples=len(preds), seconds=time.monotonic() - start, input_tokens=sum((len(r['prompt_token_ids']) for r in rs)), generated_tokens=sum((r['generated_tokens'] for r in preds)), predictions_sha256=sha(a.output / 'predictions.jsonl'), formal_test_used_for_selection=False))
    print(json.dumps(scores), flush=True)
if __name__ == '__main__':
    main()
