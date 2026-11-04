"""Materialize the TH B-cohort reserved by original pre-training protocols.

No test-informed resampling. Native evaluation remains the original frozen data.
"""
from runtime import runtime_path
import os, sys, argparse, json, collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, runtime_path('project', 'scripts/unified_language_cl'))
from common import ROOT, DATA, MODEL, sha, oh, dump, write_rows, prompt
from prepare import LOADERS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', choices=['massive', 'xnli'], required=True)
    a = ap.parse_args()
    parent = DATA / 'prepared-v1' / a.track
    p = json.loads((parent / 'protocol.json').read_text())
    A = p['training_ids']
    B = p['disjoint_reserve_ids']
    assert len(A) == len(set(A)) == len(B) == len(set(B)) == 3200 and (not set(A) & set(B))
    assert p['n_train_per_language'] == 3200 and p['steps'] == 200
    out = DATA / 'prepared-disjoint-v1' / a.track
    assert not out.exists()
    out.mkdir(parents=True)
    rs, meta = LOADERS[a.track]()
    wanted = set(B)
    chosen = {r['id']: r for r in rs if r['lang'] == 'th' and r['split'] == 'train' and (r['id'] in wanted)}
    assert set(chosen) == wanted
    del rs
    from transformers import AutoTokenizer
    from datasets import Dataset, DatasetDict
    from llamafactory.hparams import DataArguments
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    import numpy as np
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    args = DataArguments(template='qwen3', enable_thinking=False, train_on_prompt=False, cutoff_len=2048, packing=False)
    template = get_template_and_fix_tokenizer(tok, args)
    processor = SupervisedDatasetProcessor(template, tok, None, args)
    ready = []
    features = []
    counts = []
    for id in B:
        r = dict(chosen[id])
        r['prompt'] = prompt(a.track, 'th', r['text'], p['labels'])
        r['labels'] = p['labels']
        msg = [{'role': 'user', 'content': r['prompt']}, {'role': 'assistant', 'content': r['answer']}]
        source, target = template.encode_multiturn(tok, msg, None, None)[0]
        ids, labs = processor._encode_data_example(msg[:1], msg[1:], None, None, [], [], [])
        assert 0 < sum((x != -100 for x in labs[1:])) and any((x == -100 for x in labs))
        r.update(prompt_token_ids=source, input_tokens_untruncated=len(source) + len(target), assistant_tokens_untruncated=len(target), training_input_tokens=len(ids), training_supervised_tokens=sum((x != -100 for x in labs[1:])), source_truncated=len(ids) < len(source) + len(target))
        ready.append(r)
        features.append({'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labs})
        counts.append([len(source) + len(target), len(target), len(ids), r['training_supervised_tokens']])
    write_rows(out / 'th_train.jsonl', ready)
    write_rows(out / 'th_sft.jsonl', [{'messages': [{'role': 'user', 'content': r['prompt']}, {'role': 'assistant', 'content': r['answer']}]} for r in ready])
    DatasetDict({'train': Dataset.from_list(features)}).save_to_disk(str(out / 'th_tokenized'))
    dump(out / 'dataset_info.json', {'th_sft': {'file_name': 'th_sft.jsonl', 'formatting': 'sharegpt', 'columns': {'messages': 'messages'}, 'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'}}})
    arr = np.asarray(counts)
    stats = {'n': 3200, 'totals': arr.sum(0).tolist(), 'quantiles_0_50_90_95_99_100': np.percentile(arr, [0, 50, 90, 95, 99, 100], axis=0).tolist(), 'truncated': sum((r['source_truncated'] for r in ready))}
    evalfiles = {f: h for f, h in p['files'].items() if '_dev.' in f or '_test.' in f}
    derived = dict(track=a.track, status='P3_DATA_FROZEN_NOT_TRAINED', cohort='TH B disjoint from original ES/TH A IDs; B already reserved in pre-training parent protocol', model_revision=p['model_revision'], tokenizer_sha256=p['tokenizer_sha256'], parent_protocol_sha256=sha(parent / 'protocol.json'), reference_eval_protocol_sha256=sha(parent / 'protocol.json'), reference_eval_files_sha256=evalfiles, n_train_per_language=3200, steps=200, training_ids=B, original_A_ids_sha256=oh(A), B_ids_sha256=oh(B), A_B_overlap=0, token_stats_columns=p['token_stats_columns'], token_stats={'th:train': stats}, files={f: sha(out / f) for f in ['th_train.jsonl', 'th_sft.jsonl']}, labels=p['labels'], label_counts=dict(collections.Counter((r['answer'] for r in ready))), required_main_results_gate='MAIN_FIXED_FINALS_VALIDATED.json', control='Both Base->TH_B and SAME original ES_A Final->TH_B; original same-ID models reused. Fresh Adam/scheduler in each.', limits=['No new IDs selected using model scores; B is exactly the parent predeclared reserve.', 'A/B content difficulty may differ; direct TH_B control is required.', 'Language switching and content repetition remain distinct factors; not a pure language causal effect.'])
    dump(out / 'protocol.json', derived)
    dump(Path(runtime_path('configs', '')) / f'{a.track}/P3_B_COHORT_FROZEN_v1.json', {'protocol_sha256': sha(out / 'protocol.json'), 'parent_protocol_sha256': sha(parent / 'protocol.json'), 'B_matches_original_predeclared_reserve': True, 'A_B_overlap': 0, 'rows': 3200, 'steps': 200, 'training_started': False, 'native_test_reuse': True})
    print(json.dumps({'track': a.track, 'P3_data_frozen': True, 'rows': 3200, 'overlap': 0}), flush=True)
if __name__ == '__main__':
    main()
