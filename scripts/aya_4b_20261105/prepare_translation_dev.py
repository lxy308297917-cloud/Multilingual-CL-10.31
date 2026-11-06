"""Freeze a small development-only decoding comparison before observing new scores."""
from profile_paths import resolve
import json, hashlib, argparse
from pathlib import Path

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    root = Path(resolve('SECONDARY_ROOT') + '/datasets/flores200-meta-20261031/flores200_dataset/dev')
    data = {k: (root / (k + '.dev')).read_text().splitlines() for k in ['eng_Latn', 'swh_Latn']}
    assert all((len(v) == 997 for v in data.values()))
    shots = [409, 521, 656]
    ids = sorted((i for i in range(997) if i not in shots), key=lambda i: hashlib.sha256(('aya4b-mt-decoding-dev-v1|' + str(i)).encode()).hexdigest())[:64]
    expected = {'eng_Latn': '51a25f527fd81f428b6933d85a2f5e5544af92aa82042c7fbf0c4120327fba0a', 'swh_Latn': '59c8ee7cb9a702f32ff3f3e6974f4465db63df0d7172fe0d53c8bd2405daa547'}
    assert all((sha(root / (k + '.dev')) == v for k, v in expected.items()))
    rows = []
    for src, tgt, srck, tgtk in [('en', 'sw', 'eng_Latn', 'swh_Latn'), ('sw', 'en', 'swh_Latn', 'eng_Latn')]:
        template = 'Translate the following ' + ('English text into Swahili' if src == 'en' else 'Swahili text into English') + '. Output only the translation.\n\n{text}'
        messages = []
        for i in shots:
            messages.extend([{'role': 'user', 'content': template.format(text=data[srck][i])}, {'role': 'assistant', 'content': data[tgtk][i]}])
        for i in ids:
            rows.append({'id': src + '_' + tgt + '_' + str(i), 'pair_index': i, 'direction': src + '->' + tgt, 'source': data[srck][i], 'reference': data[tgtk][i], 'messages': messages + [{'role': 'user', 'content': template.format(text=data[srck][i])}]})
    path = a.output / 'development.jsonl'
    path.write_text(''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in rows)))
    cfg = {'status': 'FROZEN_BEFORE_NEW_DIAGNOSTIC_GENERATION', 'purpose': 'Development-only comparison of repetition and translation under two predefined decoding recipes; not formal results or test-based tuning', 'models': ['base', 'lora_final400'], 'directions': ['en->sw', 'sw->en'], 'samples_each_direction': 64, 'dev_indices': ids, 'fewshot_indices_excluded': shots, 'seed': 42, 'max_new_tokens': 512, 'enable_thinking': False, 'profiles': {'greedy': {'temperature': 0, 'top_p': 1, 'top_k': -1, 'min_p': 0}, 'official_non_thinking': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0}}, 'stop_token_ids': [151645, 151643], 'data_sha256': sha(path), 'dev_source_sha256': expected, 'notes': ['FLORES dev is historically exposed as a corpus; these64 indices exclude all3 fixed in-context demonstrations, are selected by deterministic hash before observing comparison outputs, and are not a new untouched confirmatory set.', 'No formal devtest scores used to choose between sampling values.', 'One fixed inference seed; stochastic scores do not establish multi-seed robustness.', 'Both Base/no-adapter and LoRA share one enable_lora engine; numerical execution differs from separate official runners, so no mixing into canonical table.'], 'official_source': 'https://huggingface.co/Qwen/Qwen3-4B#best-practices'}
    (a.output / 'manifest.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'data_sha256': cfg['data_sha256'], 'no_fewshot_overlap': not set(ids) & set(shots)}))
if __name__ == '__main__':
    main()
