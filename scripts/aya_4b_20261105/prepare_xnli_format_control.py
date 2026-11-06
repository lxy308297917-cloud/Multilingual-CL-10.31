"""Small development-only0/3shot check of the identified first-label preference."""
from profile_paths import resolve
import argparse, json, hashlib, collections
from pathlib import Path
import yaml

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def rank(i):
    return hashlib.sha256(('aya4b-xnli-format-dev-v1|' + str(i)).encode()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(exist_ok=False)
    tasks = a.output / 'tasks'
    tasks.mkdir()
    data = Path(resolve('DATA') + '/supplemental-v1')
    english = [json.loads(x) for x in (data / 'xnli_en_validation.jsonl').open()]
    bylabel = {k: sorted([int(x['id']) for x in english if x['label'] == k and int(x['id']) not in [0, 1]], key=rank) for k in range(3)}
    demos = [bylabel[k][0] for k in range(3)]
    targets = sorted([i for k in range(3) for i in bylabel[k][1:65]], key=rank)
    assert len(targets) == 192 and (not set(targets) & set(demos))
    metadata = []
    for lang in ['sw', 'en']:
        src = data / ('xnli_' + lang + '_validation.jsonl')
        rows = {int(x['id']): x for x in map(json.loads, src.open())}
        test = a.output / (lang + '_development192.jsonl')
        train = a.output / (lang + '_demonstrations3.jsonl')
        test.write_text(''.join((json.dumps(rows[i], ensure_ascii=False) + '\n' for i in targets)))
        train.write_text(''.join((json.dumps(rows[i], ensure_ascii=False) + '\n' for i in demos)))
        for shots in [0, 3]:
            name = 'aya4b_xnli_' + lang + '_format_dev_' + str(shots) + 'shot'
            cfg = {'task': name, 'dataset_path': 'json', 'dataset_kwargs': {'data_files': {'train': str(train), 'test': str(test)}}, 'training_split': 'train', 'fewshot_split': 'train', 'fewshot_config': {'sampler': 'first_n'}, 'test_split': 'test', 'output_type': 'multiple_choice', 'doc_to_text': '{{prompt}}', 'doc_to_choice': ['A', 'B', 'C'], 'doc_to_target': 'label', 'num_fewshot': shots, 'metric_list': [{'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True}], 'metadata': {'version': '1.0'}}
            (tasks / (name + '.yaml')).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        metadata.append({'language': lang, 'source_development_sha256': sha(src), 'target_file': test.name, 'target_file_sha256': sha(test), 'demonstration_file': train.name, 'demonstration_file_sha256': sha(train)})
    m = {'status': 'FROZEN_BEFORE_NEW_CONTROL_SCORING', 'purpose': 'Determine whether0shot vs3balancedexamples materially changes first-letter bias and Base-to-FFT development differences; not an alternative selected test score', 'models': ['base', 'fft_final400'], 'languages': ['sw', 'en'], 'target_examples_per_language': 192, 'examples_per_label': 64, 'demonstration_indices_in_label_order': demos, 'target_indices': targets, 'excluded_prior_engineering_smoke_indices': [0, 1], 'num_fewshot': [0, 3], 'seed': 42, 'batch_size': 4, 'source': 'XNLI validation only; no official test used for choice of examples', 'files': metadata, 'limitations': ['One fixed demonstration set and one development sample; no proof of robustness across prompts or demonstration sets', 'Both0shot and3shot must be retained regardless of outcomes; no choosing the better column as a new formal score', 'Original5010test results remain unchanged.']}
    (a.output / 'manifest.json').write_text(json.dumps(m, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'target_examples_each_language': 192, 'demonstrations': demos, 'models': ['base', 'fft'], 'four_tasks_per_model': True}))
if __name__ == '__main__':
    main()
