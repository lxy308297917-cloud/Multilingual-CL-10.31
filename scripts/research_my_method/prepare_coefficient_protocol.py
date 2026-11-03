"""Freeze task-format-aligned exploratory cohorts before any model output."""
import argparse, collections, hashlib, json
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer

PREFIX = ('Answer the multiple-choice question. Select the best answer and '
          'respond with only A, B, C, or D.\n\n')

def encode(tokenizer, row, rotation=0):
    choices = row['choices'][rotation:] + row['choices'][:rotation]
    text = PREFIX + row['question'] + '\n\n' + '\n'.join(
        key + '. ' + value for key, value in zip('ABCD', choices))
    ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    return {'input_ids': ids, 'gold': ('ABCD'.index(row['answer'])-rotation) % 4}

def main():
    p = argparse.ArgumentParser()
    for name in ['dev', 'english_source', 'base', 'output']:
        p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    rows = [json.loads(line) for line in a.dev.open()]
    manifest = json.loads((a.dev.parent/'manifest.json').read_text())
    assert hashlib.sha256(a.dev.read_bytes()).hexdigest() == manifest['paired_dev_sha256']
    assert hashlib.sha256(a.english_source.read_bytes()).hexdigest() == manifest['source_hashes']['en']
    categories = {r['sample_id']: r['subject_category'] for r in pq.read_table(
        a.english_source, columns=['sample_id', 'subject_category']).to_pylist()}
    by = {(r['semantic_id'], r['language']): r for r in rows}
    tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
    letters = [tok.encode(x, add_special_tokens=False) for x in 'ABCD']
    assert all(len(x) == 1 for x in letters) and len({x[0] for x in letters}) == 4
    langs = ['en', 'sw', 'ha']
    seed = 'semantic-coefficient-pilot-v2-20261103'
    fit = sorted({r['semantic_id'] for r in rows if r['split'] == 'fit'},
                 key=lambda sid: hashlib.sha256((seed+'|'+sid).encode()).hexdigest())
    groups = collections.defaultdict(list)
    encoded = {}
    for sid in fit:
        values = {(lang, rot): encode(tok, by[sid, lang], rot)
                  for lang in langs for rot in range(4)}
        if all(len(x['input_ids']) <= 512 for x in values.values()):
            groups[categories[sid]].append(sid)
            for (lang, rot), value in values.items():
                encoded[f'{sid}|{lang}|{rot}'] = value
    # This locked source has SIX categories, including Medical and Business.
    # Freeze the actual source taxonomy instead of inventing a four-way map.
    assert set(groups) == {'STEM', 'Humanities', 'Social Sciences', 'Medical', 'Business', 'Other'}
    m = next((m for m in [3] if all(len(v) >= m for v in groups.values())), None)
    assert m is not None
    fit_ids = [sid for cat in sorted(groups) for sid in groups[cat][:m]]
    engineering_fit = [sid for cat in sorted(groups) for sid in groups[cat][:2]]
    engineering_eval = [groups[cat][2] for cat in sorted(groups)]
    confirmation = sorted({r['semantic_id'] for r in rows if r['split'] == 'confirmation'})
    excluded = []
    for sid in confirmation:
        values = {lang: encode(tok, by[sid, lang]) for lang in langs}
        if any(len(x['input_ids']) > 1024 for x in values.values()):
            excluded.append(sid)
        else:
            for lang, value in values.items():
                encoded[f'{sid}|{lang}|0'] = value
    confirmation = [sid for sid in confirmation if sid not in excluded]
    keep = set(fit_ids + engineering_fit + engineering_eval + confirmation)
    encoded = {k: v for k, v in encoded.items() if k.rsplit('|', 2)[0] in keep}
    protocol = {
        'version': 2, 'seed': seed, 'languages': langs,
        'fit_ids': fit_ids, 'categories': {sid: categories[sid] for sid in keep},
        'fit_per_category': m, 'option_rotations': [0, 1, 2, 3],
        'engineering_fit_ids': engineering_fit, 'engineering_eval_ids': engineering_eval,
        'confirmation_ids': confirmation, 'excluded_overlong_confirmation_ids': excluded,
        'answer_token_ids': [x[0] for x in letters], 'fit_max_tokens': 512,
        'confirmation_max_tokens': 1024, 'norm_fractions': [.1, .3],
        'conditions': ['raw', 'block_shared', 'block_shuffled', 'block_random',
                       'reuse_shared', 'reuse_shuffled', 'reuse_random'],
        'gradient': 'Mean of four option-rotation four-way CE coefficient gradients per item; no per-row normalization',
        'shared_space': 'Within-category centered EN/SW symmetric coefficient cross-covariance; up to2 positive eigenvectors per matrix',
        'shuffled': 'Within-category fixed cyclic derangement of SW IDs; same rank per matrix as paired, top algebraic eigenvectors',
        'active_subset': 'Fit-only: 0<paired rank<coefficient dimension and all seven raw/projected directions nondegenerate at relative1e-8. Same active modules for every condition; no noise fallback',
        'norm_matching': 'Global10% or30% of acquired projected update norm on active modules; every matrix matches raw-gradient norm allocation; all other weights fixed',
        'scoring': '0shot Qwen3 nonthinking; FP32 master/eager; argmax of A/B/C/D next-token logits. ACC primary, four-way CE auxiliary',
        'historical_exposure': manifest['known_prior_exposure'],
        'semantic_translation_review': False, 'formal_test_used': False,
        'adoption_claim_allowed': False, 'frozen_before_model_outputs': True,
        'paired_dev_sha256': manifest['paired_dev_sha256'], 'encoded': encoded}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in protocol.items() if k != 'encoded'}, ensure_ascii=False))

if __name__ == '__main__':
    main()
