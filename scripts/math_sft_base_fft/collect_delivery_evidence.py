"""Curate real accounting and answer-free per-item flags for the research handoff.

Run only after all route-A roles are complete. No raw prompts, answers, generated
programs, host process identities or weights are exported.
"""
from profile_paths import resolve
import csv, gzip, hashlib, io, json, sys, time, re
from pathlib import Path
R = Path(resolve('ROOT'))
C = Path(resolve('CHECKPOINTS'))
sys.path.insert(0, str(R / 'scripts/math_sft_base_fft'))
from aggregate_route_a import general_flags
from common import LANGS, DATA

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def emit(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')

def rows(p):
    with p.open() as f:
        for s in f:
            if s.strip():
                yield json.loads(s)

def stable_id(s):
    return hashlib.sha256(str(s).encode()).hexdigest()

def export_flags(out):
    columns = ['language', 'epoch', 'task', 'item_id_sha256', 'identity_sha256', 'correct', 'parsed', 'strict_correct', 'generated_tokens', 'finish_reason']
    files = []
    base_predictions = [x for i in [1, 2, 3] for x in rows(R / f'results/math_sft_base_fft/base-census-worker{i}-v1/predictions.jsonl')]
    for lang in LANGS:
        for ep in [0, 1, 2] if lang in ['sw', 'bn', 'ja'] else [0]:
            if ep:
                d = R / f'results/math_sft_base_fft/A/{lang}/epoch{ep}'
                targets = list(rows(d / 'math/predictions.jsonl'))
                poly = list(rows(d / 'polymath/scored.jsonl'))
            else:
                targets = [x for x in base_predictions if x['lang'] == lang]
                poly = list(rows(R / f'results/math_sft_base_fft/base-census-polymath-v1/{lang}-score/scored.jsonl'))
            name = f'target_{lang}_E{ep}.csv.gz'
            p = out / name
            records = []
            for x in targets:
                if x['benchmark'] not in ['mgsm_rev2', 'msvamp']:
                    continue
                records.append(dict(language=lang, epoch=ep, task=x['benchmark'], item_id_sha256=stable_id(x['id']), identity_sha256=x['prompt_token_sha256'], correct=int(x['correct']), parsed=int(x['parsed'] is not None), strict_correct=int(x['strict_correct']), generated_tokens=x['generated_tokens'], finish_reason=x['finish_reason']))
            target_by_id = {x['id']: x for x in targets}
            for x in poly:
                x = dict(target_by_id[x['id']], **x)
                records.append(dict(language=lang, epoch=ep, task='polymath_' + x['level'], item_id_sha256=stable_id(x['id']), identity_sha256=x.get('prompt_token_sha256', ''), correct=int(x['correct']), parsed=int(x['box_parse']), strict_correct='', generated_tokens=x.get('generated_tokens', ''), finish_reason=x.get('finish_reason', '')))
            assert len(records) == 1750
            tasks = sorted({r['task'] for r in records})
            diagnostics = {}
            for task in tasks:
                rs = [r for r in records if r['task'] == task]
                source_rs = [x for x in targets if ('polymath_' + x['level'] if x['benchmark'] == 'polymath' else x['benchmark']) == task]
                assert len(source_rs) == len(rs)
                diagnostics[task] = {'n': len(rs), 'correct': sum((r['correct'] for r in rs)), 'unparsed': sum((not r['parsed'] for r in rs)), 'parsed_but_answer_incorrect': sum((r['parsed'] and (not r['correct']) for r in rs)), 'correct_but_not_strict_match': None if task.startswith('polymath_') else sum((r['correct'] and (not r['strict_correct']) for r in rs)), 'hit_length_cap': sum((r['finish_reason'] == 'length' for r in rs)), 'unparsed_and_cap': sum((not r['parsed'] and r['finish_reason'] == 'length' for r in rs)), 'correct_and_cap': sum((r['correct'] and r['finish_reason'] == 'length' for r in rs)), 'generated_tokens': sum((r['generated_tokens'] for r in rs)), 'strict_correct': None if task.startswith('polymath_') else sum((r['strict_correct'] for r in rs)), 'has_hash_final_marker': sum(('####' in x['output'] for x in source_rs)), 'has_box_command': sum((chr(92) + 'boxed' in x['output'] for x in source_rs)), 'hash_marker_without_box_command': sum(('####' in x['output'] and chr(92) + 'boxed' not in x['output'] for x in source_rs))}
            emit(out / f'target_{lang}_E{ep}_diagnostics.json', {'language': lang, 'epoch': ep, 'tasks': diagnostics, 'scope': 'Measured termination, parsing and correctness intersections; cap association alone does not prove wrong answers would be correct with a larger budget.'})
            write_csv_gz(p, columns, records)
            files.append(dict(file=name, rows=len(records), sha256=sha(p)))
    for role, folder in [('Base', R / 'results/math_sft_base_fft/base-general-v1')] + [(f'{l}-E{e}', R / f'results/math_sft_base_fft/A/{l}/epoch{e}/general') for l in ['sw', 'bn', 'ja'] for e in [1, 2]]:
        flags = general_flags(folder)
        records = []
        for task, items in flags.items():
            for k, v in sorted(items.items()):
                records.append(dict(role=role, task=task, item_id_sha256=stable_id(k), identity_sha256=stable_id(json.dumps(v['identity'], sort_keys=True)), correct=int(v['correct'])))
        name = f'general_{role}.csv.gz'
        p = out / name
        write_csv_gz(p, ['role', 'task', 'item_id_sha256', 'identity_sha256', 'correct'], records)
        files.append(dict(file=name, rows=len(records), sha256=sha(p)))
    emit(out / 'manifest.json', {'scope': 'Answer-free per-item correctness and diagnostic flags. IDs hashed consistently; source/prompt identity retained. Raw questions, references, model outputs and generated programs omitted. General metrics are the four predeclared primary metrics; full auxiliary scalar metrics in all_roles.csv.', 'files': files})

def write_csv_gz(p, columns, records):
    with p.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=columns)
                w.writeheader()
                w.writerows(records)

def export_run_manifests(out, result):
    profiles = [(str(C), '${CHECKPOINTS}'), (str(DATA), '${DATA}'), (resolve('MODEL'), '${MODEL}'), (str(R), '${ROOT}'), (resolve('LOGS'), '${LOGS}'), (resolve('SANDBOX_WORKSPACE'), '${SANDBOX_WORKSPACE}'), (resolve('ENV_ROOT'), '${ENV_ROOT}'), (resolve('DATASET_ROOT'), '${DATASET_ROOT}'), (resolve('USER_STORAGE_ROOT'), '${USER_STORAGE_ROOT}'), (resolve('SECONDARY_STORAGE_ROOT'), '${SECONDARY_STORAGE_ROOT}')]

    def scrub(x):
        if isinstance(x, str):
            for a, b in profiles:
                x = x.replace(a, b)
            return re.sub('GPU-[0-9a-fA-F-]{36}', '${GPU_UUID}', x)
        if isinstance(x, list):
            return [scrub(v) for v in x]
        if isinstance(x, dict):
            return {scrub(k): scrub(v) for k, v in x.items() if k not in ['gpu_uuid', 'gpu_uuids', 'hostname', 'pid', 'outer_created_pid']}
        return x
    sources = []
    for i in [1, 2, 3]:
        for name in ['run.json', 'GENERATION_DONE.json']:
            sources.append((f'Base-census-worker{i}/' + name, result / f'base-census-worker{i}-v1' / name))
    roles = [('Base', result / 'base-general-v1', None)] + [(f'{lang}-E{ep}', result / f'A/{lang}/epoch{ep}/general', result / f'A/{lang}/epoch{ep}/math') for lang in ['sw', 'bn', 'ja'] for ep in [1, 2]]
    for role, folder, math in roles:
        for name in ['launch.json', 'GPU_PHASE_DONE.json', 'core/TASK_DONE.json', 'mmlu/TASK_DONE.json', 'code/generation/manifest.json', 'code/generation/DONE.json', 'code/generation/cost.json', 'code-execution/scores.json']:
            sources.append((role + '/general/' + name, folder / name))
        if math:
            for name in ['run.json', 'GENERATION_DONE.json']:
                sources.append((role + '/math/' + name, math / name))
    records = []
    for name, source in sources:
        assert source.exists(), source
        dest = out / name
        emit(dest, scrub(read(source)))
        records.append({'file': name, 'actual_manifest_sha256': sha(source), 'public_manifest_sha256': sha(dest)})
    emit(out / 'manifest.json', {'files': records, 'adaptation': 'Known deployment path prefixes become declared profile keys. GPU UUID and host process/hostname fields omitted; UUID strings inside commands replaced by GPU_UUID. All other metric, source, model, protocol, version, input and output hashes retained. Public JSON is not byte-identical to private original; both hashes recorded. No raw questions, answers or generated solutions in these manifests.'})

def verify_role_protocols(result):
    log_root = Path(resolve('LOGS'))
    base = result / 'base-general-v1'
    base_launch = read(base / 'launch.json')
    base_code = read(base / 'code/generation/manifest.json')
    reference_math = read(result / 'base-census-worker1-v1/run.json')
    base_executor = read(base / 'code-execution/executor_identity.json')
    canonical = read(base / 'code-execution/canonical_calibration-results.json')
    assert len(canonical) == 164 and all((x['base_pass'] and x['plus_pass'] and (x['infrastructure_status'] == 'COMPLETE') for x in canonical))
    math_assets = {}
    proofs = []

    def engine_receipt(p, wanted):
        found = None
        with p.open(errors='replace') as f:
            for line in f:
                if 'Initializing a ' in line and 'LLM engine' in line:
                    found = line
                    break
        assert found and 'Initializing a ' + wanted + ' LLM engine' in found, (p, wanted)
        return {'engine': wanted, 'actual_log_sha256': sha(p), 'initialization_marker': found[found.index('Initializing a '):].split('with config:')[0].strip()}
    for i in [1, 2, 3]:
        name = f'base-census-worker{i}-v1'
        x = read(result / name / 'run.json')
        assert x['protocol_sha256'] == reference_math['protocol_sha256'] and x['generation'] == reference_math['generation'] and (x['weights_sha256'] == base_launch['weights_sha256'])
        math_assets.update(x['assets'])
        proofs.append({'role': name, 'weights_sha256': x['weights_sha256'], 'math_engine': engine_receipt(log_root / name / 'job.log', 'V1')})
    for role, folder, lang, ep in [('Base', base, None, 0)] + [(f'{l}-E{e}', result / f'A/{l}/epoch{e}/general', l, e) for l in ['sw', 'bn', 'ja'] for e in [1, 2]]:
        launch = read(folder / 'launch.json')
        code = read(folder / 'code/generation/manifest.json')
        weights = launch['weights_sha256']
        if ep:
            training = read(C / f'mathoctopus_{lang}_full_2epochs_seed42_v1/TRAINING_DONE.json')
            assert weights == training['epochs'][ep - 1]['weights_sha256']
            math = read(folder.parent / 'math/run.json')
            assert math['weights_sha256'] == weights and math['protocol_sha256'] == reference_math['protocol_sha256'] and (math['generation'] == reference_math['generation']) and (math['assets'] == {lang + '_eval.jsonl': math_assets[lang + '_eval.jsonl']})
        used_tasks = lambda x: {k: v for k, v in x['frozen_task_files'].items() if k.endswith('/gsm8k.yaml') or k.endswith('/ifeval.yaml') or k.startswith(str(Path(resolve('MMLU_TASKS'))) + '/')}
        assert used_tasks(launch) == used_tasks(base_launch) and len(used_tasks(launch)) >= 59 and (launch['code_manifest_sha256'] == base_launch['code_manifest_sha256'])
        assert code['benchmark_protocol'] == base_code['benchmark_protocol'] and code['backend'] == base_code['backend'] and (code['data_manifest_hash'] == base_code['data_manifest_hash']) and (code['sample_ids_sha256'] == base_code['sample_ids_sha256']) and (code['prompt_tokens_sha256'] == base_code['prompt_tokens_sha256']) and (code['batch_size'] == base_code['batch_size'])
        assert read(folder / 'code-execution/scores.json')['executor_identity'] == base_executor
        code_weights = {Path(k).name: v for k, v in code['checkpoint_identity'].items() if k.endswith('.safetensors')}
        assert code_weights == weights
        proof = {'role': role, 'weights_sha256': weights, 'same_frozen_general_tasks': True, 'same_code_data_template_decoding_backend_and_prompt_tokens': True, 'core_engine': engine_receipt(folder / 'core/run.log', 'V0'), 'code_engine': engine_receipt(folder / 'code/run.log', 'V0')}
        if ep:
            proof['math_engine'] = engine_receipt(log_root / f'A-{lang}-epoch{ep}-full-eval-v1/job.log', 'V1')
        proofs.append(proof)
    return {'status': 'PASSED_ACTUAL_ROLE_PROTOCOL_IDENTITIES', 'math_protocol_sha256': reference_math['protocol_sha256'], 'model_roles_checked': 7, 'base_census_workers_checked': 3, 'canonical_references_passed': 164, 'canonical_result_sha256': sha(base / 'code-execution/canonical_calibration-results.json'), 'executor_identity': base_executor, 'polymath_equivalence_calibration': read(result / 'polymath-calibration-v3/CALIBRATION_DONE.json'), 'proofs': proofs, 'scope': 'Actual logs, data, prompt, task, decoding/backend manifests and training/evaluation weight hashes compared across all fixed roles. Task-level paired analysis additionally checks each source/prompt/target hash; code output equivalence is not assumed.'}

def main():
    result = R / 'results/math_sft_base_fft'
    assert (result / 'routeA-analysis-v1/DONE.json').exists(), 'Full analysis not complete'
    out = result / 'delivery-evidence-v1'
    assert not out.exists(), 'Independent output required'
    out.mkdir()
    cost = []
    input_evidence = []
    for lang in ['sw', 'bn', 'ja']:
        name = f'mathoctopus_{lang}_full_2epochs_seed42_v1'
        p = C / name
        x = read(p / 'TRAINING_DONE.json')
        assert x['status'] == 'COMPLETE'
        emit(out / f'{lang}_training_validation.json', x)
        for filename in ['training_data.json', 'fingerprint_source_ids.json', 'dataset_info.json']:
            original = DATA / 'training-v1' / lang / filename
            payload = original.read_bytes()
            assert b'/data0/users/' not in payload and b'/data1/' not in payload
            if filename == 'training_data.json':
                assert sha(original) == x['data_manifest_sha256']
            destination = out / 'data_inputs' / lang / (filename + '.gz')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('wb') as stream:
                with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
                    compressed.write(payload)
            input_evidence.append({'file': str(destination.relative_to(out)), 'uncompressed_sha256': sha(original), 'compressed_sha256': sha(destination), 'exact_original_bytes_after_decompression': True})
        budget = R / 'configs/math_sft_base_fft' / (name + '.budget.json')
        emit(out / 'data_inputs' / lang / 'budget.json', read(budget))
        metrics = read(p / 'train_results.json')
        emit(out / f'{lang}_train_results.json', metrics)
        state = read(p / 'trainer_state.json')
        history = state.get('log_history', [])
        metric_keys = ['step', 'epoch', 'loss', 'learning_rate', 'grad_norm', 'train_loss', 'train_runtime', 'train_samples_per_second', 'train_steps_per_second']
        (out / f'{lang}_optimizer_metrics.jsonl').write_text(''.join((json.dumps({k: v for k, v in row.items() if k in metric_keys}) + '\n' for row in history)))
        rank_memory = []
        for rank in [0, 1]:
            account = read(p / f'accounting-rank{rank}.json')
            keep = ['step', 'resume_initial_step', 'local_counts', 'global_counts', 'elapsed_this_process_seconds', 'max_memory_allocated', 'max_memory_reserved', 'max_rss_kib', 'parameters', 'world_size', 'exposure_file_sha256', 'trace_rows', 'trace_covers_all_local_samples']
            emit(out / f'{lang}_accounting_rank{rank}.json', {k: account[k] for k in keep if k in account})
            rank_memory.append({'rank': rank, 'max_torch_allocated_bytes': account['max_memory_allocated'], 'max_torch_reserved_bytes': account['max_memory_reserved'], 'max_rss_kib': account['max_rss_kib']})
        logs = list(rows(p / 'trainer_log.jsonl'))
        allowed = ['current_steps', 'total_steps', 'loss', 'lr', 'epoch', 'percentage', 'elapsed_time', 'remaining_time']
        (out / f'{lang}_training_progress.jsonl').write_text(''.join((json.dumps({k: r[k] for k in allowed if k in r}) + '\n' for r in logs)))
        epoch_costs = []
        for epoch_record in x['epochs']:
            checkpoint = p / f"checkpoint-{epoch_record['steps']}"
            records = [read(checkpoint / f'accounting-rank{rank}.json') for rank in [0, 1]]
            elapsed = max((a['elapsed_this_process_seconds'] for a in records))
            epoch_costs.append({'epoch': epoch_record['epoch'], 'steps': epoch_record['steps'], 'cumulative_input_tokens': epoch_record['cumulative_input_tokens'], 'cumulative_supervised_tokens': epoch_record['cumulative_shifted_supervised_tokens'], 'cumulative_exposures': epoch_record['cumulative_samples'], 'cumulative_callback_elapsed_seconds': elapsed, 'cumulative_two_gpu_hours_to_checkpoint': elapsed * 2 / 3600})
        e = x['epochs'][-1]
        seconds = metrics['train_runtime']
        cost.append({'language': lang, 'source_records_per_epoch': e['source_records_per_epoch'], 'epochs': 2, 'optimizer_steps': e['steps'], 'epoch1_optimizer_steps': x['epochs'][0]['steps'], 'epoch2_optimizer_steps': e['steps'], 'epoch1_supervised_tokens': x['epochs'][0]['cumulative_shifted_supervised_tokens'], 'actual_exposures': e['cumulative_samples'], 'input_tokens': e['cumulative_input_tokens'], 'supervised_tokens': e['cumulative_shifted_supervised_tokens'], 'trainer_runtime_seconds': seconds, 'assigned_training_gpus': 2, 'per_rank_peak_memory': rank_memory, 'epoch_checkpoint_costs': epoch_costs, 'trainer_gradient_norm_log_records': sum(('grad_norm' in row for row in history)), 'trainer_gpu_hours': seconds * 2 / 3600, 'actual_input_tokens_per_second': e['cumulative_input_tokens'] / seconds, 'actual_supervised_tokens_per_second': e['cumulative_shifted_supervised_tokens'] / seconds})
    emit(out / 'data_inputs/manifest.json', {'inputs': input_evidence, 'scope': 'Original frozen training metadata and input-label/source-ID fingerprints; compressed exactly. No raw question, answer, token sequence or host path.'})
    audit = read(DATA / 'mathoctopus-audit-v1/audit.json')
    emit(out / 'data_inputs/mathoctopus_all_language_audit.json', audit)
    phases = []
    for role, p in [('Base-general', result / 'base-general-v1/GPU_PHASE_DONE.json')] + [(f'{l}-E{e}', result / f'A/{l}/epoch{e}/GPU_PHASE_DONE.json') for l in ['sw', 'bn', 'ja'] for e in [1, 2]]:
        x = read(p)
        phases.append({'role': role, 'status': x['status'], 'receipt_sha256': sha(p), 'reported_seconds': x.get('seconds'), 'gpu_count': 1})
    queue = read(R / 'configs/math_sft_base_fft/route_a_queue_state_v2.json')
    assert queue['status'] == 'GPU_PHASE_COMPLETE'
    job_costs = [{'job': name, 'kind': job['kind'], 'assigned_gpu_count': len(job['gpu_uuids']), 'allocation_wall_seconds': job['finished'] - job['started'], 'allocation_gpu_hours': (job['finished'] - job['started']) * len(job['gpu_uuids']) / 3600, 'exit_code': job['exit_code']} for name, job in queue['jobs'].items()]
    census = [dict(worker=i, **read(result / f'base-census-worker{i}-v1/GENERATION_DONE.json')) for i in [1, 2, 3]]
    engineering = read(C / 'engineering33_sw_seed42_v1/train_results.json')
    emit(out / 'cost_accounting.json', {'training': cost, 'route_A_allocation_jobs': job_costs, 'route_A_total_allocation_gpu_hours': sum((x['allocation_gpu_hours'] for x in job_costs)), 'nine_language_base_generation': census, 'engineering_validation_trainer_runtime_seconds': engineering['train_runtime'], 'gpu_evaluation_phases': phases, 'training_total_trainer_gpu_hours': sum((x['trainer_gpu_hours'] for x in cost)), 'scope': 'Trainer runtimes include trainer checkpoint activity; multiplied by two assigned GPUs. Evaluation phase wall time is reported separately, may contain CPU loading/waits and does not measure kernel active time. Nine-language Base census, engineering validation, audit CPU and queue reservation overhead are additional costs, not included in trainer-only total. FLOP counter from stock trainer is not treated as reliable measured device FLOPs.'})
    export_flags(make_dir(out / 'per_item_flags'))
    export_run_manifests(make_dir(out / 'actual_run_manifests'), result)
    emit(out / 'protocol_identity_verification.json', verify_role_protocols(result))
    emit(out / 'DONE.json', {'status': 'COMPLETE', 'generated': time.time(), 'answer_free_results': True, 'source_analysis_done_sha256': sha(result / 'routeA-analysis-v1/DONE.json')})
    print(json.dumps({'status': 'COMPLETE', 'files': sum((x.is_file() for x in out.rglob('*'))), 'directory': str(out)}))

def make_dir(p):
    p.mkdir()
    return p
if __name__ == '__main__':
    main()
