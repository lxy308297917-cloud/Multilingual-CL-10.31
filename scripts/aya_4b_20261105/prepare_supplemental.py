"""Freeze XNLI SW/EN and XCOPA SW with a common instruction/option-ACC protocol."""
from profile_paths import resolve
import argparse, json, hashlib, collections
from pathlib import Path
import pyarrow.parquet as pq

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--xcopa', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    taskdir = a.output / 'tasks'
    taskdir.mkdir()
    root = Path(resolve('SECONDARY_ROOT') + '/datasets/unified-language-cl-20261104/sources/facebook--xnli')
    metadata = json.loads((root / 'metadata.json').read_text())
    assert metadata['sha'] == 'b8dd5d7af51114dbda02c0e3f6133f332186418e'
    sources = {}
    specs = []

    def write(name, split, rows):
        out = a.output / (name + '_' + split + '.jsonl')
        out.write_text(''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in rows)))
        return out

    def task(name, n, splitfiles):
        cfg = {'task': 'aya4b_' + name, 'dataset_path': 'json', 'dataset_kwargs': {'data_files': {k: str(v) for k, v in splitfiles.items()}}, 'output_type': 'multiple_choice', 'test_split': 'test', 'validation_split': 'validation', 'doc_to_text': '{{prompt}}', 'doc_to_choice': list('ABC'[:n]), 'doc_to_target': 'label', 'num_fewshot': 0, 'metric_list': [{'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True}], 'metadata': {'version': '1.0'}}
        import yaml
        (taskdir / (name + '.yaml')).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        cfg['task'] = 'aya4b_' + name + '_dev'
        cfg['test_split'] = 'validation'
        (taskdir / (name + '_dev.yaml')).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        specs.append({'task': name, 'options': n, 'splits': {k: {'file': v.name, 'sha256': sha(v), 'rows': sum((1 for _ in v.open()))} for k, v in splitfiles.items()}, 'task_yaml_sha256': sha(taskdir / (name + '.yaml'))})
    for lang in ['sw', 'en']:
        paths = {}
        for split, expected in [('test', 5010), ('validation', 2490)]:
            source = root / 'all_languages' / (split + '-00000-of-00001.parquet')
            xs = pq.read_table(source).to_pylist()
            assert len(xs) == expected
            selected = []
            sources[str(source.relative_to(root))] = sha(source)
            for i, x in enumerate(xs):
                premise = x['premise'][lang]
                h = dict(zip(x['hypothesis']['language'], x['hypothesis']['translation']))[lang]
                assert x['label'] in range(3)
                prompt = 'Determine the relation between the premise and the hypothesis.\n\nPremise: ' + premise + '\nHypothesis: ' + h + '\n\nA. Entailment\nB. Neutral\nC. Contradiction\nAnswer:'
                selected.append({'id': str(i), 'prompt': prompt, 'label': x['label'], 'premise_cluster_sha256': hashlib.sha256(premise.encode()).hexdigest()})
            paths[split] = write('xnli_' + lang, split, selected)
        task('xnli_' + lang, 3, paths)
    paths = {}
    for split, expected in [('test', 500), ('validation', 100)]:
        source = a.xcopa / (split + '-00000-of-00001.parquet')
        xs = pq.read_table(source).to_pylist()
        assert len(xs) == expected
        selected = []
        sources['xcopa/sw/' + source.name] = sha(source)
        for i, x in enumerate(xs):
            assert x['label'] in range(2) and x['question'] in ['cause', 'effect']
            question = 'What was the cause?' if x['question'] == 'cause' else 'What happened as a result?'
            prompt = x['premise'] + '\n' + question + '\n\nA. ' + x['choice1'] + '\nB. ' + x['choice2'] + '\nAnswer:'
            selected.append({'id': str(x['idx']), 'prompt': prompt, 'label': x['label'], 'premise_cluster_sha256': hashlib.sha256(x['premise'].encode()).hexdigest()})
        paths[split] = write('xcopa_sw', split, selected)
    task('xcopa_sw', 2, paths)
    manifest = {'status': 'FROZEN_BEFORE_MODEL_SCORING', 'task_family': 'Supplemental instruction-format multiple-choice likelihood ACC; not the original completion-style Harness XNLI prompt', 'models': ['base', 'fft_final400', 'lora_final400', 'ssu_final400'], 'num_fewshot': 0, 'batch_size': 4, 'seed': 42, 'model_revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'primary_metric': 'ACC', 'source_revisions': {'facebook/xnli': metadata['sha'], 'cambridgeltl/xcopa': '042f78955ba48e6404616762fa6e05e839c3907a'}, 'source_file_sha256': sources, 'tasks': specs, 'notes': ['English relation labels and instructions held constant across EN/SW; source sentences retain native benchmark language. XCOPA cause/effect instruction uses the same English wording across future languages.', 'Development validation split only for two-example engineering smoke and any later recipe development; official test never chooses labels/prompts/checkpoint.', 'English causal reference omitted: no XCOPA-en exists; do not fabricate parallel English results.', 'All three tasks and all four models will be reported even if scores decline; no selection on test gains.', 'One training seed; source XNLI historically available to prior research, exposure recorded.', 'Use original labels0=entailment,1=neutral,2=contradiction; XCOPA0/1. Models score A/B/C letter likelihoods, not generated answer parsing.']}
    dump(a.output / 'manifest.json', manifest)
    print(json.dumps({'tasks': [(x['task'], x['splits']['test']['rows']) for x in specs], 'frozen_manifest_sha256': sha(a.output / 'manifest.json')}))
if __name__ == '__main__':
    main()
