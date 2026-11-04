"""Correct primary summary scores from immutable completed generation records.

No repeated generation, model selection, prompt edits or reference clipping.
"""
from runtime import runtime_path
import argparse, json, collections, os, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from common import ROOT, sha, dump, rows
from summary_scoring import score_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert (a.source / 'DONE.json').exists()
    assert not a.output.exists()
    rs = rows(a.source / 'predictions.jsonl')
    source_hash = sha(a.source / 'predictions.jsonl')
    assert source_hash == json.loads((a.source / 'DONE.json').read_text())['predictions_sha256']
    a.output.mkdir(parents=True)
    started = time.monotonic()
    source_scores = json.loads((a.source / 'scores.json').read_text())
    scored = []
    scores = {}
    for lang in sorted({r['lang'] for r in rs}):
        group = [r for r in rs if r['lang'] == lang]
        vals = score_rows([r['prediction'] for r in group], [r['answer'] for r in group], lang)
        key = next((k for k in source_scores if k.startswith(lang + ':')))
        original = source_scores[key]
        for metric in ['rouge1', 'rouge2']:
            assert abs(sum((x[metric] for x in vals)) / len(vals) - original[metric]) < 1e-08, (lang, metric)
        scores[key] = dict(original, score=sum((x['rougeLsum'] for x in vals)) / len(vals), **{m: sum((x[m] for x in vals)) / len(vals) for m in ['rouge1', 'rouge2', 'rougeLsum']})
        scored.extend(({'id': r['id'], 'lang': lang, **v} for r, v in zip(group, vals)))
    (a.output / 'predictions.jsonl').symlink_to(a.source.resolve() / 'predictions.jsonl')
    run = json.loads((a.source / 'run.json').read_text())
    run['original_generation_run_sha256'] = sha(a.source / 'run.json')
    run['primary_scorer_version'] = 'v2_explicit_multilingual_sentence_tokenizer'
    run['primary_scorer_sha256'] = sha(Path(__file__).with_name('summary_scoring.py'))
    run['score_correction_protocol_sha256'] = sha(Path(runtime_path('configs', '')) / 'SUMMARY_SCORING_CORRECTION_v2.json')
    dump(a.output / 'run.json', run)
    (a.output / 'per_sample_scores.jsonl').write_text(''.join((json.dumps(v) + '\n' for v in scored)))
    dump(a.output / 'scores.json', scores)
    dump(a.output / 'DONE.json', {'status': 'DONE_CORRECTED_PRIMARY_SCORING', 'cpu_wall_seconds': time.monotonic() - started, 'additional_gpu_generation_tokens': 0, 'samples': len(rs), 'predictions_sha256': source_hash, 'original_predictions_unchanged': True, 'no_generation_rerun': True, 'original_buggy_scores_preserved': True, 'rouge1_2_regression_verified': True, 'scorer_sha256': run['primary_scorer_sha256'], 'score_correction_protocol_sha256': run['score_correction_protocol_sha256']})
    print(json.dumps({'source': str(a.source), 'corrected_scores': scores}), flush=True)
if __name__ == '__main__':
    main()
