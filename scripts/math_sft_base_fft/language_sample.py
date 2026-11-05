"""Bounded language-presence diagnostic only; no training-data filtering."""
from profile_paths import resolve
import os, collections, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from common import *
import fasttext
model_path = Path(resolve('GLOTLID'))
model = fasttext.load_model(str(model_path))
expected = {'sw': 'swh_Latn', 'bn': 'ben_Beng', 'th': 'tha_Thai', 'ja': 'jpn_Jpan', 'zh': 'cmn_Hani', 'ru': 'rus_Cyrl', 'fr': 'fra_Latn', 'de': 'deu_Latn', 'es': 'spa_Latn'}
result = {}
for lang in LANGS:
    source = DATA / 'mathoctopus-audit-v1' / f'{lang}_all_valid.jsonl'
    sample = sorted(rows(source), key=lambda r: oh([42, 'bounded-language', r['id']]))[:32]
    values = []
    for r in sample:
        for side in ['question', 'answer']:
            text = r[side].replace('\n', ' ')
            if sum((c.isalpha() for c in text)) < 50:
                values.append({'id': r['id'], 'side': side, 'measured': False})
                continue
            labels, probs = model.predict(text[:6000], k=3)
            pred = labels[0].replace('__label__', '')
            values.append({'id': r['id'], 'side': side, 'measured': True, 'top1': pred, 'top1_probability': float(probs[0]), 'matches_expected': pred == expected[lang]})
    stats = {}
    for side in ['question', 'answer']:
        known = [x for x in values if x['side'] == side and x['measured']]
        stats[side] = {'sampled': 32, 'measured': len(known), 'matching_expected': sum((x['matches_expected'] for x in known)), 'top1_counts': dict(collections.Counter((x['top1'] for x in known)))}
    result[lang] = {'source_sha256': sha(source), 'sample_ids_sha256': oh([r['id'] for r in sample]), 'stats': stats, 'items': values}
    print(lang, stats, flush=True)
out = ROOT / 'results/math_sft_base_fft/training-language-sample-v1.json'
assert not out.exists()
dump(out, {'status': 'COMPLETE', 'sampled_records_per_language': 32, 'model_sha256': sha(model_path), 'languages': result, 'role': 'SUPPLEMENTARY_LANGUAGE_PRESENCE_ONLY', 'not_used_to_change_frozen_data': True, 'limitations': 'Automatic LID is not semantic or reasoning-quality validation, and cannot certify translation correctness. Short/math-heavy texts excluded from measured count. No native speaker review claimed.'})
