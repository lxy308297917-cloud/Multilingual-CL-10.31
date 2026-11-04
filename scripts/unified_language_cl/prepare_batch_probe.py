"""Freeze a length-stress cohort of 64 real NLI training examples, no dev/test."""
from runtime import runtime_path
import json
from pathlib import Path
import yaml
from common import ROOT, DATA, MODEL, sha, oh, dump, rows, write_rows

def main():
    from datasets import Dataset, DatasetDict, load_from_disk
    source = DATA / 'prepared-v1/xnli'
    data = load_from_disk(str(source / 'th_tokenized'))['train']
    rs = rows(source / 'th_train.jsonl')
    indexes = sorted(range(len(data)), key=lambda i: (-len(data[i]['input_ids']), rs[i]['id']))[:64]
    out = DATA / 'engineering/nli-long64-v1'
    assert not out.exists()
    out.mkdir(parents=True)
    features = [data[i] for i in indexes]
    selected = [rs[i] for i in indexes]
    DatasetDict({'train': Dataset.from_list(features)}).save_to_disk(str(out / 'th_tokenized'))
    write_rows(out / 'train64.jsonl', [{'messages': [{'role': 'user', 'content': r['prompt']}, {'role': 'assistant', 'content': r['answer']}]} for r in selected])
    dump(out / 'dataset_info.json', {'train64': {'file_name': 'train64.jsonl', 'formatting': 'sharegpt', 'columns': {'messages': 'messages'}, 'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'}}})
    dump(out / 'cohort.json', dict(scope='ENGINEERING_ONLY_NOT_RESEARCH_DATA_SELECTION', source_protocol_sha256=sha(source / 'protocol.json'), selection='64 longest encoded TH examples from pre-frozen NLI train3200, no dev/test', source_ids=[r['id'] for r in selected], input_lengths=[len(r['input_ids']) for r in features], input_label_multiset_sha256=oh(sorted((oh([r['input_ids'], r['labels']]) for r in features)))))
    template = yaml.safe_load((Path(runtime_path('configs', '')) / 'mathoctopus/mathoctopus_single_es_seed42_v1.yaml').read_text())
    for micro in [2, 4]:
        cfg = dict(template)
        name = f'xnli_long64_micro{micro}_engineering_v1'
        cfg.update(dataset_dir=str(out), dataset='train64', tokenized_path=str(out / 'th_tokenized'), per_device_train_batch_size=micro, gradient_accumulation_steps=16 // (2 * micro), save_only_model=True, output_dir=runtime_path('checkpoints', '/') + name, logging_dir=runtime_path('logs', '/') + name)
        p = Path(runtime_path('configs', '')) / 'xnli' / f'{name}.yaml'
        assert not p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(p, flush=True)
if __name__ == '__main__':
    main()
