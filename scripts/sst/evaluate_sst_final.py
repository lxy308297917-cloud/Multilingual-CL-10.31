"""One finished SST Final, existing frozen benchmark drivers, no model selection.

The bench-v2 generator's `method=full` denotes a full weight export; this
wrapper supplies the actual SST identity. Its results are isolated from FFT.
"""
import argparse, collections, datetime, hashlib, json, os, subprocess, sys, time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--training-record', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--method', choices=['sst_norefresh', 'sst_r10'], required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    r = a.project.resolve()
    base = Path('/local/models/Qwen3-1.7B')
    trainpy = '/local/envs/qwen3-multilingual-20261031/bin/python'
    evalpy = '/local/envs/qwen3-eval-vllm-20261101/bin/python'
    sys.path.insert(0, str(r/'scripts'))
    from run_checkpoint_eval_17b import build_command, task_spec
    logs = Path('/local/logs/qwen3-multilingual/sst-evaluation')/a.name
    result = r/'results/sst'/a.name
    core = logs/'core'
    model_args = (f'pretrained={a.model},tokenizer={base},dtype=bfloat16,max_model_len=8192,'
                  'gpu_memory_utilization=0.85,max_num_seqs=32,max_num_batched_tokens=8192,'
                  'enable_prefix_caching=True,seed=42,enforce_eager=True,enable_thinking=False')
    core_tasks = 'sft_v1_gsm8k,sft_v1_ifeval,sft_v1_mgsm_native_cot_sw,sft_v1_xifeval_sw'
    commands = [{'task': 'core', 'command': [evalpy, str(r/'scripts/sft_v1/harness.py'),
         'run', '--model', 'vllm', '--model_args', model_args, '--tasks', core_tasks,
         '--include_path', str(r/'configs/sft_v1/eval_tasks'), '--apply_chat_template',
         '--batch_size', 'auto', '--output_path', str(core/'results'), '--log_samples', '--seed', '42']}]
    for task in ['belebele', 'mmlu', 'gmmlu']:
        commands.append({'task': task, 'command': build_command(task, a.model, 'full', base, logs/task/'results')})
    # Match the completed v2 SUM and full1012 MT protocols, not the older500 MT.
    # Infer the real frozen dataset locations from the completed Base manifests.
    dataset_paths = {}
    for name in ['xlsum_sw_v2', 'xlsum_en_v2', 'flores200_en_sw_full_devtest_v2', 'flores200_sw_en_full_devtest_v2']:
        cfg = json.loads((r/'configs/bench_v2'/f'{name}.json').read_text())
        candidates = list(Path('/local/datasets').glob(f'*/{name}/manifest.json'))
        candidates += list(Path('/local/datasets').glob(f'*/*/{name}/manifest.json'))
        candidates = [path for path in candidates if json.loads(path.read_text()).get('data_sha256') == cfg['data_sha256']]
        unique = {path.resolve() for path in candidates}
        assert len(unique) == 1, (name, list(unique))
        dataset_paths[name] = next(iter(unique)).parent
        commands.append({'task': name, 'command': [evalpy, str(r/'scripts/bench_v2/generate.py'),
            '--dataset', str(dataset_paths[name]), '--model', str(a.model), '--tokenizer', str(base),
            '--method', 'full', '--gpu-uuid', a.gpu_uuid, '--output', str(logs/name/'generation'),
            '--scope', 'full', '--batch-size', '32', '--max-user-gpus', '4']})
    spec = {'actual_method': a.method, 'artifact_kind_for_legacy_driver': 'full',
            'model': str(a.model), 'training_record': str(a.training_record), 'commands': commands,
            'scope': 'Fixed400-step Final, all tasks reported; no score threshold/selection',
            'gpu_uuid': a.gpu_uuid, 'formal_test_used_for_selection': False}
    if a.dry_run:
        print(json.dumps(spec, indent=2)); return
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == a.gpu_uuid and ',' not in a.gpu_uuid
    assert not result.exists() and not logs.exists()
    result.mkdir(parents=True); logs.mkdir(parents=True)
    start = time.time()
    def dump(name, value):
        (result/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    env = dict(os.environ, HF_HOME='/local/hf-cache-qwen3', HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
        NLTK_DATA='/local/datasets/qwen3-nltk-data-20261031',
        VLLM_USE_V1='0', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    def execute(path, cmd, extra=None):
        with path.open('x') as out:
            subprocess.run(list(map(str, cmd)), cwd=r, env=dict(env, **(extra or {})),
                           stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, check=True)
    try:
        done = json.loads((a.training_record/'COMPLETED.json').read_text())
        assert done['exit_code'] == 0 and done['global_step'] == 400
        assert Path(done['output']).resolve() == a.model.resolve()
        trainer = json.loads((a.model/'trainer_state.json').read_text())
        assert trainer['global_step'] == trainer['max_steps'] == 400
        count = json.loads((a.model/'accounting-rank0.json').read_text())['global_counts']
        assert (count['samples'], count['input_tokens'], count['shifted_assistant_tokens']) == (6400, 2884423, 1078347)
        original = json.loads((base/'config.json').read_text())
        actual = json.loads((a.model/'config.json').read_text())
        for key in ['model_type', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                    'num_attention_heads', 'num_key_value_heads', 'vocab_size', 'tie_word_embeddings']:
            assert original.get(key) == actual.get(key), key
        assert (a.model/'tokenizer.json').is_file()
        assert json.loads((a.model/'generation_config.json').read_text())['eos_token_id'] == json.loads((base/'generation_config.json').read_text())['eos_token_id']
        weights = sorted(a.model.glob('*.safetensors')); assert weights
        index = a.model/'model.safetensors.index.json'
        if index.exists():
            assert set(json.loads(index.read_text())['weight_map'].values()) == {x.name for x in weights}
        files = [Path(__file__), r/'scripts/run_checkpoint_eval_17b.py',
                 r/'scripts/sft_v1/harness.py', r/'scripts/sft_v1/vllm_compat.py',
                 r/'scripts/bench_v2/generate.py', r/'scripts/bench_v2/common.py',
                 r/'scripts/bench_v2/prompting.py', r/'scripts/bench_v2/score.py',
                 r/'scripts/bench_v2/scoring.py']
        files += list((r/'configs/sft_v1/eval_tasks').glob('*.yaml'))
        files += [r/'configs/bench_v2'/f'{name}.json' for name in dataset_paths]
        frozen = {str(x): sha(x) for x in files}
        spec.update(started_at=datetime.datetime.now().astimezone().isoformat(), wrapper_pid=os.getpid(),
                    weights_sha256={x.name: sha(x) for x in weights}, source_files_sha256=frozen,
                    training_receipt_sha256=sha(a.training_record/'COMPLETED.json'))
        dump('launch.json', spec)
        execute(logs/'base-verified.txt', [trainpy, r/'scripts/verify_model_lock.py', r/'configs/model_17b.lock.json', base])
        assets = sorted({asset for task in ['belebele', 'mmlu', 'gmmlu', 'ifeval', 'gsm8k'] for asset in task_spec(task)[3]})
        execute(logs/'assets-verified.txt', [trainpy, r/'scripts/verify_assets.py', *assets])
        stack = json.loads((r/'configs/sft_v1/stack.lock.json').read_text())
        for name, digest in stack['tasks'].items():
            assert sha(r/'configs/sft_v1/eval_tasks'/name) == digest, name
        for component in ['harness', 'benchmax']:
            identity = stack[component]
            assert subprocess.check_output(['git', '-C', identity['source'], 'rev-parse', 'HEAD'], text=True).strip() == identity['commit']
        source_lock = json.loads(Path('/local/datasets/qwen3-sft-v1-20261101/benchmarks/sources.lock.json').read_text())
        for name in ['juletxara/mgsm', 'LLaMAX/BenchMAX_Rule-based']:
            source = source_lock[name]
            for file, digest in source['files'].items():
                assert sha(Path(source['root'])/file) == digest, (name, file)
        execute(logs/'train-environment.freeze.txt', [trainpy, '-m', 'pip', 'freeze'])
        execute(logs/'eval-environment.freeze.txt', [evalpy, '-m', 'pip', 'freeze'])
        tasks_root = Path('/local/datasets/qwen3-eval-tasks-20261031')
        for name, digest in json.loads((tasks_root/'manifest.json').read_text())['file_sha256'].items():
            assert sha(tasks_root/name) == digest
        expected_core = json.loads((Path('/local/logs/qwen3-multilingual/sft-v1')/'evaluation_environment_preflight.json').read_text())['tasks']
        completed = []
        for entry in commands:
            task = entry['task']; folder = logs/task; folder.mkdir()
            assert all(sha(path) == digest for path, digest in frozen.items()), 'Frozen driver changed'
            execute(folder/'gpu-preflight.json', [trainpy, r/'scripts/preflight_gpu.py', '--uuid', a.gpu_uuid,
                    '--min-free-mib', '21000', '--max-user-gpus', '4', '--settle-seconds', '30'])
            extra = {'SFT_VLLM_TRACE': str(core/'completion-metadata.jsonl')} if task == 'core' else None
            execute(folder/'eval.log', entry['command'], extra)
            if task in ['belebele', 'mmlu', 'gmmlu']:
                prefix, total, n = {'belebele': ('belebele_', 1800, 2), 'mmlu': ('mmlu_', 14042, 57),
                                    'gmmlu': ('global_mmlu_full_sw_', 14042, 57)}[task]
                execute(folder/'accuracy-audit.json', [trainpy, r/'scripts/audit_harness_accuracy.py',
                    '--run-dir', folder, '--task-prefix', prefix, '--expected-tasks', n, '--expected-total', total])
            elif task == 'core':
                summaries = list(folder.glob('results/**/results_*.json')); assert len(summaries) == 1
                summary = json.loads(summaries[0].read_text()); audit = {}
                expected = [x for x in expected_core if not x['task'].startswith('sft_v1_mmlu_prox_')]
                assert len(expected) == 4
                for item in expected:
                    name, n = item['task'], item['rows']; saved = summary['results'][name]
                    assert saved['sample_len'] == n
                    paths = list(summaries[0].parent.glob(f'samples_{name}_*.jsonl')); assert len(paths) == 1
                    by_filter = collections.defaultdict(list)
                    for line in paths[0].open():
                        row = json.loads(line); by_filter[row['filter']].append(row)
                    scores = {}
                    for label, rows in by_filter.items():
                        assert len(rows) == len({x['doc_id'] for x in rows}) == n
                        for metric in rows[0]['metrics']:
                            values = [v for row in rows for v in (row[metric] if isinstance(row[metric], list) else [row[metric]])]
                            assert values and all(v in (0, 1) for v in values)
                            rate = sum(values)/len(values)
                            assert abs(rate-saved[f'{metric},{label}']) < 1e-10
                            scores[f'{metric},{label}'] = {'rate': rate, 'n': len(values)}
                    audit[name] = scores
                trace = [json.loads(x) for x in (core/'completion-metadata.jsonl').open()]
                assert len(trace) == sum(x['rows'] for x in expected)
                (folder/'accuracy-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
            else:
                execute(folder/'score.log', [evalpy, r/'scripts/bench_v2/score.py', '--run',
                         folder/'generation', '--output', folder/'scores'])
                assert json.loads((folder/'scores/summary.json').read_text())['run_status'] == 'VERIFIED_COMPLETE'
            completed.append(task)
            dump('progress.json', {'completed': completed, 'remaining': [x['task'] for x in commands if x['task'] not in completed],
                                   'elapsed_seconds': time.time()-start})
        dump('DONE.json', {'actual_method': a.method, 'tasks': completed, 'exit_code': 0,
                           'wall_seconds': time.time()-start, 'formal_test_used_for_selection': False})
    except BaseException as error:
        dump('FAILED.json', {'error': repr(error), 'wall_seconds': time.time()-start})
        raise


if __name__ == '__main__':
    main()
