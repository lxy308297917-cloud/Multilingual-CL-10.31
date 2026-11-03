"""Extend existing fixed MCQ protocols to official Hausa, without test selection."""
import argparse, collections, hashlib, json, shutil, zipfile
from pathlib import Path
import pyarrow.parquet as pq
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ['project', 'raw', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    r = a.project
    assets = json.loads((r/'assets.lock.json').read_text())['assets']
    archive = Path(assets['belebele_official_zip']['path'])
    assert sha(archive) == assets['belebele_official_zip']['sha256']
    with zipfile.ZipFile(archive) as z:
        content = z.read('Belebele/hau_Latn.jsonl')
    ha = a.output/'belebele_hau_Latn.jsonl'; ha.write_bytes(content)
    rows = [json.loads(x) for x in ha.open()]
    assert len(rows) == 900
    for lang in ['eng_Latn', 'swh_Latn']:
        source = assets['belebele_'+lang]
        assert sha(source['path']) == source['sha256']
        other = [json.loads(x) for x in open(source['path'])]
        assert len(other) == 900
        for x, y in zip(rows, other):
            for key in ['link', 'question_number', 'correct_answer_num']:
                assert x[key] == y[key], (lang, key)
    task = {'include': str(r/'configs/eval_tasks/_belebele_locked_base.yaml'),
            'task': 'belebele_hau_Latn_sequential_locked',
            'dataset_kwargs': {'data_files': {'test': str(ha)}}}
    (a.output/'belebele_ha.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    rev = '0e619dbeb34206cd48705a1a0ea7fb21cae09993'
    expected = {'dev': 'f76830fc89623ab26b9f3150aa90688ee8aee75e5795d13e2302724744fd25ae',
                'test': 'b17d14b2fe9bf848935145ad49db494850930f23627880ce14d0f88e09e84766'}
    paths = {split: a.raw/f'ha-{split}.parquet' for split in expected}
    data = {}
    for split, path in paths.items():
        assert sha(path) == expected[split]
        data[split] = pq.read_table(path).to_pylist()
        assert len(data[split]) == (285 if split == 'dev' else 14042)
        ids = [x['sample_id'] for x in data[split]]
        assert len(ids) == len(set(ids))
        assert len({x['subject'] for x in data[split]}) == 57
        assert all(x['question'].strip() and x['answer'] in 'ABCD' and all(
            x['option_'+letter].strip() for letter in 'abcd') for x in data[split])
        old = pq.read_table(assets['global_mmlu_sw_'+split]['path']).to_pylist()
        assert [(x['sample_id'], x['subject'], x['answer']) for x in data[split]] == [
            (x['sample_id'], x['subject'], x['answer']) for x in old]
    old_root = Path('/local/datasets/qwen3-eval-tasks-20261031')
    old_manifest = json.loads((old_root/'manifest.json').read_text())
    for name, digest in old_manifest['file_sha256'].items():
        assert sha(old_root/name) == digest
    source = old_root/'global_mmlu_full_sw'; target = a.output/'global_mmlu_full_ha'
    target.mkdir()
    source_hashes = {}
    for file in source.iterdir():
        if not file.is_file():
            continue
        source_hashes[file.name] = sha(file)
        if file.name == '_sw_template_yaml':
            cfg = yaml.safe_load(file.read_text())
            cfg['dataset_path'] = 'parquet'; cfg['dataset_name'] = None
            cfg['dataset_kwargs'] = {'data_files': {k: str(v) for k, v in paths.items()}}
            (target/'_ha_template_yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        elif file.suffix == '.yaml':
            name = file.name.replace('_sw', '_ha')
            (target/name).write_text(file.read_text().replace('_sw', '_ha'))
        elif file.name == 'utils.py':
            shutil.copyfile(file, target/file.name)
    assert len(list(target.glob('global_mmlu_full_ha_*_locked.yaml'))) == 57
    files = {str(path.relative_to(a.output)): sha(path) for path in a.output.rglob('*') if path.is_file()}
    manifest = {'purpose': 'Formal reporting of fixed sequential FFT stages; no test-based selection',
        'language_selected_before_test_scores': True, 'target_language': 'ha',
        'belebele': {'archive_sha256': assets['belebele_official_zip']['sha256'],
                     'member': 'Belebele/hau_Latn.jsonl', 'rows': 900, 'shots': 0,
                     'primary': 'acc', 'secondary': 'acc_norm', 'EN_SW_ID_label_alignment': True},
        'global_mmlu': {'repository': 'CohereLabs/Global-MMLU', 'revision': rev,
                       'files': {k: {'path': str(v), 'sha256': expected[k]} for k, v in paths.items()},
                       'rows': 14042, 'subjects': 57, 'shots': 5,
                       'fewshot_sampler': 'first_n within each subject dev', 'primary': 'acc',
                       'SW_ID_subject_label_alignment': True},
        'source_task_files_sha256': source_hashes, 'files_sha256': files,
        'backend': 'Same HF harness and Base tokenizer as original SW/EN ACC;BF16,batch4 for Belebele,batch1 for Global-MMLU',
        'semantic_translation_independently_reviewed': False,
        'source_library_versions': 'Original locked task utils/templates retained;only language/data/task names extended'}
    (a.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': 'READY_NOT_EVALUATED', 'belebele': 900, 'global_mmlu': 14042, 'subjects': 57}))


if __name__ == '__main__':
    main()
