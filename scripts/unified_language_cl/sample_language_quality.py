"""Bounded post-preparation language diagnostic; never changes frozen data."""
from runtime import runtime_path
import collections, hashlib, json, sys
from pathlib import Path
DATA = Path(runtime_path('data', 'prepared-v1'))
OUT = Path(runtime_path('results', 'language_sample_v1.json'))

def main():
    import fasttext
    model = fasttext.load_model(runtime_path('glotlid', 'model_v3.bin'))
    results = {}
    expected = {'es': 'spa_Latn', 'th': 'tha_Thai', 'fr': 'fra_Latn', 'en': 'eng_Latn'}
    for task in ['mathoctopus', 'xlsum']:
        for lang in expected:
            rs = [json.loads(l) for l in (DATA / task / f'{lang}_train.jsonl').open()]
            sample = sorted(rs, key=lambda r: hashlib.sha256(('lid42:' + r['id']).encode()).hexdigest())[:64]
            values = []
            for r in sample:
                text = r['answer'].replace('\n', ' ')
                if sum((c.isalpha() for c in text)) < 50:
                    values.append({'id_sha256': hashlib.sha256(r['id'].encode()).hexdigest(), 'measured': False})
                    continue
                labels, probs = model.predict(text[:6000], k=3)
                pred = labels[0].replace('__label__', '')
                values.append({'id_sha256': hashlib.sha256(r['id'].encode()).hexdigest(), 'measured': True, 'top1': pred, 'top1_probability': float(probs[0]), 'matches_expected': pred == expected[lang]})
            known = [r for r in values if r['measured']]
            results[task + ':' + lang] = {'sampled': 64, 'measured': len(known), 'matching_expected': sum((r['matches_expected'] for r in known)), 'top1_counts': dict(collections.Counter((r['top1'] for r in known))), 'items': values}
    result = {'scope': 'SUPPLEMENTARY_POST_PREPARATION_TRAIN_ONLY_DIAGNOSTIC', 'selection': 'SHAseed64 from each pre-frozen train3200; assistant target only, under50letters unmeasured', 'not_used_to_change_frozen_data': True, 'limitations': 'Automatic LID cannot certify semantic translation or answer correctness; no complete native-speaker review is claimed. Not a downstream benchmark or method-adoption signal.', 'results': results}
    assert not OUT.exists()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: {kk: v[kk] for kk in ['sampled', 'measured', 'matching_expected', 'top1_counts']} for k, v in results.items()}, ensure_ascii=False))
if __name__ == '__main__':
    main()
