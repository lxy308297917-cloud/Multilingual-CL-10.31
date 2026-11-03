"""Fixed full-test reporting for SST-NoRefresh SW -> Hausa, never selection."""
import argparse, collections, datetime, hashlib, json, os, subprocess, sys, time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--partition', choices=['main', 'old', 'ha', 'source-ha', 'all'], default='main')
    a = p.parse_args(); r = a.project
    sys.path.insert(0, str(r/'scripts'))
    from run_checkpoint_eval_17b import build_command
    py = '/local/envs/qwen3-multilingual-20261031/bin/python'
    evalpy = '/local/envs/qwen3-eval-vllm-20261101/bin/python'
    base = Path('/local/models/Qwen3-1.7B')
    sw = Path('/local/checkpoints/sst_sw_aya6400_20261102/sw_aya6400_sst_r99_norefresh_seed42_final_v1')
    ha = Path('/local/checkpoints/multilingual_cl_sequential_20261103/sw_ha_6400x2_sst_norefresh_seed42_final_v1')
    taskroot = Path('/local/datasets/multilingual-cl-sequential-20261103/formal-ha-bench-v1')
    logs = Path('/local/logs/qwen3-multilingual/sequential-evaluation')/('sst-norefresh-v1-'+a.partition)
    output = r/'results/research_my_method'/('sequential-sst-norefresh-formal-v1-'+a.partition)
    entries = []
    def original(task):
        name = 'stage2-'+task
        return {'name': name, 'model': str(ha), 'task': task,
                'command': build_command(task, ha, 'full', base, logs/name/'results')}
    entries.append(original('belebele'))
    for stage, model in [('Base', base), ('SW-SST-NoRefresh', sw), ('SW-HA-SST-NoRefresh', ha)]:
        for task in ['ha-belebele', 'ha-gmmlu']:
            name = stage+'-'+task
            task_name = 'belebele_hau_Latn_sequential_locked' if task == 'ha-belebele' else 'global_mmlu_full_ha_locked'
            command = [str(Path(py).parent/'lm-eval'), 'run', '--model', 'hf', '--model_args',
                f'pretrained={model},tokenizer={base},dtype=bfloat16,enable_thinking=False',
                '--tasks', task_name, '--include_path', str(taskroot), '--num_fewshot',
                '0' if task == 'ha-belebele' else '5', '--apply_chat_template', '--batch_size',
                '4' if task == 'ha-belebele' else '1', '--device', 'cuda:0', '--output_path',
                str(logs/name/'results'), '--log_samples', '--seed', '42']
            entries.append({'name': name, 'model': str(model), 'task': task, 'command': command})
    entries += [original('gmmlu'), original('mmlu')]
    name = 'stage2-core'; model_args = (
        f'pretrained={ha},tokenizer={base},dtype=bfloat16,max_model_len=8192,'
        'gpu_memory_utilization=0.85,max_num_seqs=32,max_num_batched_tokens=8192,'
        'enable_prefix_caching=True,seed=42,enforce_eager=True,enable_thinking=False')
    entries.append({'name': name, 'model': str(ha), 'task': 'core', 'command': [evalpy,
        str(r/'scripts/sft_v1/harness.py'), 'run', '--model', 'vllm', '--model_args', model_args,
        '--tasks', 'sft_v1_gsm8k,sft_v1_ifeval,sft_v1_mgsm_native_cot_sw,sft_v1_xifeval_sw',
        '--include_path', str(r/'configs/sft_v1/eval_tasks'), '--apply_chat_template',
        '--batch_size', 'auto', '--output_path', str(logs/name/'results'), '--log_samples', '--seed', '42']})
    # Finish all reading comparisons before longer knowledge tasks.
    entries.sort(key=lambda x: (0 if 'belebele' in x['task'] else 1 if x['task'] != 'core' else 2))
    if a.partition != 'all':
        entries = [x for x in entries if (
            x['model'] != str(ha) if a.partition == 'source-ha' else x['model'] == str(ha))]
    if a.partition in ['old','ha']:
        entries=[x for x in entries if x['task'].startswith('ha-')==(a.partition=='ha')]
    plan = {'purpose': 'Formal fixed-stage reporting, no test-based decisions', 'entries': entries,
            'max_user_gpus': 4, 'actual_method': 'SST-NoRefresh', 'stage_budgets': '6400 SW then6400 HA;tokens not matched',
            'formal_test_used_for_selection': False, 'all_predeclared_tasks_reported': True,
            'execution_partition': a.partition, 'partition_rule': 'main=Stage2 all6 tasks;old=4 EN/SW tasks;ha=2 HA tasks;old andha disjoint union equalsmain;source-ha references separate'}
    if a.dry_run:
        print(json.dumps(plan, indent=2)); return
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == a.gpu_uuid and ',' not in a.gpu_uuid
    assert not output.exists() and not logs.exists()
    output.mkdir(parents=True); logs.mkdir(parents=True); start = time.time()
    env = dict(os.environ, HF_HOME='/local/hf-cache-qwen3', HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', VLLM_USE_V1='0',
        VLLM_WORKER_MULTIPROC_METHOD='spawn', NLTK_DATA='/local/datasets/qwen3-nltk-data-20261031')
    def dump(name, value): (output/name).write_text(json.dumps(value, indent=2)+'\n')
    def run(file, cmd, extra=None):
        with file.open('x') as log:
            subprocess.run(list(map(str, cmd)), cwd=r, env=dict(env, **(extra or {})),
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        receipt = r/'results/research_my_method/sw_ha_6400x2_sst_norefresh_seed42_final_v1/COMPLETED.json'
        done = json.loads(receipt.read_text()); assert done['exit_code'] == 0 and done['global_step'] == 400
        assert Path(done['output']).resolve() == ha.resolve()
        for model in [sw, ha]:
            state = json.loads((model/'trainer_state.json').read_text())
            assert state['global_step'] == state['max_steps'] == 400
        manifest = json.loads((taskroot/'manifest.json').read_text())
        for file, digest in manifest['files_sha256'].items(): assert sha(taskroot/file) == digest
        for source in manifest['global_mmlu']['files'].values(): assert sha(source['path']) == source['sha256']
        sourcefiles = [Path(__file__), r/'scripts/run_checkpoint_eval_17b.py', r/'scripts/sft_v1/harness.py',
            r/'scripts/sft_v1/vllm_compat.py', r/'configs/sft_v1/stack.lock.json', taskroot/'manifest.json']
        sourcefiles += list((r/'configs/eval_tasks').glob('*.yaml'))
        sourcefiles += list((r/'configs/sft_v1/eval_tasks').rglob('*.yaml'))
        frozen = {str(path): sha(path) for path in sourcefiles}
        plan.update(started_at=datetime.datetime.now().astimezone().isoformat(), wrapper_pid=os.getpid(),
            source_sha256=frozen, model_weight_sha256={str(model): {p.name: sha(p) for p in sorted(model.glob('*.safetensors'))}
                                                     for model in [base, sw, ha]}, training_receipt_sha256=sha(receipt))
        dump('launch.json', plan)
        run(logs/'base-lock.txt', [py, r/'scripts/verify_model_lock.py', r/'configs/model_17b.lock.json', base])
        run(logs/'assets-check.txt', [py, r/'scripts/verify_assets.py', 'belebele_eng_Latn', 'belebele_swh_Latn',
             'mmlu_dev', 'mmlu_test', 'global_mmlu_sw_dev', 'global_mmlu_sw_test', 'ifeval_input', 'gsm8k_train', 'gsm8k_test'])
        stack = json.loads((r/'configs/sft_v1/stack.lock.json').read_text())
        for path, digest in stack['tasks'].items(): assert sha(r/'configs/sft_v1/eval_tasks'/path) == digest
        for component in ['harness', 'benchmax']:
            identity = stack[component]
            assert subprocess.check_output(['git', '-C', identity['source'], 'rev-parse', 'HEAD'], text=True).strip() == identity['commit']
        source_lock = json.loads(Path('/local/datasets/qwen3-sft-v1-20261101/benchmarks/sources.lock.json').read_text())
        for name in ['juletxara/mgsm', 'LLaMAX/BenchMAX_Rule-based']:
            for file, digest in source_lock[name]['files'].items(): assert sha(Path(source_lock[name]['root'])/file) == digest
        run(logs/'train-environment.freeze.txt', [py, '-m', 'pip', 'freeze'])
        run(logs/'eval-environment.freeze.txt', [evalpy, '-m', 'pip', 'freeze'])
        old_tasks = Path('/local/datasets/qwen3-eval-tasks-20261031')
        for path, digest in json.loads((old_tasks/'manifest.json').read_text())['file_sha256'].items():
            assert sha(old_tasks/path) == digest
        core_expected = [x for x in json.loads(Path('/local/logs/qwen3-multilingual/sft-v1/evaluation_environment_preflight.json').read_text())['tasks']
                         if not x['task'].startswith('sft_v1_mmlu_prox_')]
        completed = []
        for entry in entries:
            assert all(sha(path) == digest for path, digest in frozen.items())
            folder = logs/entry['name']; folder.mkdir()
            run(folder/'gpu-preflight.json', [py, r/'scripts/preflight_gpu.py', '--uuid', a.gpu_uuid,
                 '--min-free-mib', '21000', '--max-user-gpus', '4', '--settle-seconds', '30'])
            extra = {'SFT_VLLM_TRACE': str(folder/'completion-metadata.jsonl')} if entry['task'] == 'core' else None
            run(folder/'eval.log', entry['command'], extra)
            task = entry['task']
            if task != 'core':
                prefix, n, count = {'belebele': ('belebele_', 1800, 2),
                    'ha-belebele': ('belebele_hau_', 900, 1), 'ha-gmmlu': ('global_mmlu_full_ha_', 14042, 57),
                    'gmmlu': ('global_mmlu_full_sw_', 14042, 57), 'mmlu': ('mmlu_', 14042, 57)}[task]
                run(folder/'accuracy-audit.json', [py, r/'scripts/audit_harness_accuracy.py', '--run-dir', folder,
                    '--task-prefix', prefix, '--expected-total', n, '--expected-tasks', count])
            else:
                summaries = list(folder.glob('results/**/results_*.json')); assert len(summaries) == 1
                results = json.loads(summaries[0].read_text())['results']; audit = {}
                for spec in core_expected:
                    name, n = spec['task'], spec['rows']; assert results[name]['sample_len'] == n
                    samples = list(summaries[0].parent.glob(f'samples_{name}_*.jsonl')); assert len(samples) == 1
                    groups = collections.defaultdict(list)
                    for line in samples[0].open():
                        row = json.loads(line); groups[row['filter']].append(row)
                    values = {}
                    for label, rows in groups.items():
                        assert len(rows) == len({x['doc_id'] for x in rows}) == n
                        for metric in rows[0]['metrics']:
                            flags = [v for row in rows for v in (row[metric] if isinstance(row[metric], list) else [row[metric]])]
                            assert flags and all(v in (0, 1) for v in flags)
                            rate = sum(flags)/len(flags); assert abs(rate-results[name][f'{metric},{label}']) < 1e-10
                            values[f'{metric},{label}'] = {'rate': rate, 'n': len(flags)}
                    audit[name] = values
                (folder/'accuracy-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
            completed.append(entry['name'])
            dump('progress.json', {'completed': completed, 'total': len(entries), 'seconds': time.time()-start})
        dump('DONE.json', {'status': 'FORMAL_REPORT_COMPLETE', 'completed': completed,
                           'seconds': time.time()-start, 'formal_test_used_for_selection': False})
    except BaseException as error:
        dump('FAILED.json', {'error': repr(error), 'seconds': time.time()-start}); raise


if __name__ == '__main__':
    main()
