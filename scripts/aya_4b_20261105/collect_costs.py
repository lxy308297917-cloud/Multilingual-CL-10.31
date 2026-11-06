"""Actual training/evaluation wall times and sampled resource coverage, no kernel-time claims."""
from profile_paths import resolve
import argparse, json, datetime, subprocess, collections, hashlib
from pathlib import Path
LOG = Path(resolve('LOGS'))
CKPT = Path(resolve('CHECKPOINTS'))
DATA = Path(resolve('DATA'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dt(s):
    return datetime.datetime.fromisoformat(s)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    training = json.loads((a.training_audit / 'training_cost.json').read_text())
    assert set(training) == {'fft', 'lora', 'ssu'}
    evaluations = []
    raw = []
    for role in ['base', 'fft', 'lora', 'ssu']:
        for group in ['belebele', 'gmmlu', 'mmlu']:
            root = LOG / (role + '-harness-v1') / group
            assert (root / 'DONE.json').exists()
            files = list(root.glob('results/**/results_*.json'))
            assert len(files) == 1
            result = json.loads(files[0].read_text())
            seconds = float(result['total_evaluation_time_seconds'])
            assert seconds > 0
            evaluations.append({'model': role, 'task_group': group, 'wall_seconds': seconds, 'measurement': 'Official Harness total_evaluation_time_seconds', 'source_sha256': sha(files[0])})
        root = LOG / (role + '-generation-v2')
        assert (root / 'DONE.json').exists()
        for name in ['flores200_en_sw_full_devtest_v2', 'flores200_sw_en_full_devtest_v2', 'xlsum_sw_v2', 'xlsum_en_v2']:
            f = root / name / 'cost.json'
            v = json.loads(f.read_text())
            evaluations.append({'model': role, 'task_group': name, 'wall_seconds': v['elapsed_seconds'], 'input_tokens': v['input_tokens'], 'generated_tokens': v['generated_tokens'], 'measurement': 'Generation driver elapsed including engine load/generation, excluding prior CPU prompt/hash prep', 'source_sha256': sha(f)})
        f = list((root / 'core/results').glob('**/results_*.json'))
        assert len(f) == 1
        v = json.loads(f[0].read_text())
        evaluations.append({'model': role, 'task_group': 'GSM8K+IFEval', 'wall_seconds': float(v['total_evaluation_time_seconds']), 'measurement': 'Official Harness total_evaluation_time_seconds', 'source_sha256': sha(f[0])})
    peaks = collections.defaultdict(int)
    rss = collections.defaultdict(int)
    sampled = collections.Counter()
    sample_sources = {}
    for name in ['resource-samples-v1.jsonl', 'resource-samples-v2.jsonl']:
        f = LOG / name
        if not f.exists():
            continue
        with f.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                x = json.loads(line)
                sampled[name] += 1
                for g in x['gpu_processes']:
                    peaks[g['role']] = max(peaks[g['role']], g['memory_mib'])
                for process in x['processes'].values():
                    rss[process['command_role']] = max(rss[process['command_role']], process['rss_kib'])
        sample_sources[name] = sha(f)
    engineering = []
    for role, version in [('lora', 'v1'), ('fft', 'v2'), ('ssu', 'v2')]:
        folder = CKPT / (role + '_smoke2_' + version)
        done = json.loads((folder / 'SMOKE_DONE.json').read_text())
        engineering.append({'method': role, 'scope': 'Isolated two-step engineering smoke, never benchmark-selected', 'wall_seconds': (dt(done['ended_at']) - dt(done['started_at'])).total_seconds(), 'samples': done['samples'], 'input_tokens': done['exposure']['input_tokens'], 'assistant_tokens': done['exposure']['shifted_assistant_tokens']})
    probe = LOG / 'lora-device-diagnostic-v1/cost.json'
    if probe.exists():
        v = json.loads(probe.read_text())
        engineering.append({'method': 'lora', 'scope': 'Four-example physical GPU identity/memory recovery probe; not model selection', 'wall_seconds': v['elapsed_seconds'], 'samples': 4, 'input_tokens': v['input_tokens'], 'generated_tokens': v['generated_tokens'], 'source_sha256': sha(probe)})
    failed = []
    for d in LOG.glob('*-pipeline-v*'):
        if not (d / 'FAILED.json').exists():
            continue
        start = json.loads((d / 'run-start.json').read_text())
        end = json.loads((d / 'FAILED.json').read_text())
        failed.append({'attempt': d.name, 'wall_seconds': (dt(end['ended_at']) - dt(start['started_at'])).total_seconds(), 'reason': 'CPUAdam JIT PATH/ninja missing' if d.name == 'fft-pipeline-v1' else 'SSU mask JSON passed as a directory', 'failed_log_sha256': sha(d / 'smoke.log'), 'restarted_in_independent_directory': True})
    superseded = []
    for role in ['base', 'lora']:
        old = LOG / (role + '-generation-v1')
        for f in sorted(old.glob('*/cost.json')):
            v = json.loads(f.read_text())
            superseded.append({'model': role, 'attempt': 'generation-v1', 'task_group': f.parent.name, 'wall_seconds': v['elapsed_seconds'], 'generated_tokens': v['generated_tokens'], 'source_sha256': sha(f), 'included_in_canonical_comparison': False})
        for f in sorted(old.glob('core/results/**/results_*.json')):
            v = json.loads(f.read_text())
            superseded.append({'model': role, 'attempt': 'generation-v1', 'task_group': 'GSM8K+IFEval', 'wall_seconds': float(v['total_evaluation_time_seconds']), 'source_sha256': sha(f), 'included_in_canonical_comparison': False})
    failure_logs = []
    for label, rel in [('LoRA v1 summary prefill OOM', 'lora-generation-v1/xlsum_sw_v2.log'), ('LoRA v2 initial memory report inconsistency', 'lora-generation-v2-failed-initial/flores200_en_sw_full_devtest_v2.log')]:
        f = LOG / rel
        if f.exists():
            failure_logs.append({'attempt': label, 'log_sha256': sha(f), 'wall_seconds': None, 'measurement': 'Failure has no completed driver cost record; exact wall time not inferred or reported as zero'})
    mask = json.loads((DATA / 'ssu-mask-v1').read_text())
    ms = json.loads((LOG / 'ssu-mask-v1/run-start.json').read_text())
    mask_seconds = (dt(mask['created_utc']) - dt(ms['started_at'])).total_seconds()
    sizes = {}
    for label, path in [('new_data', DATA), ('new_checkpoints_including_preserved_smokes', CKPT), ('new_logs', LOG)]:
        sizes[label] = int(subprocess.check_output(['du', '-sb', str(path)], text=True).split()[0])
    result = {'status': 'COMPLETE_ACTUAL_COST_ACCOUNTING', 'training': training, 'evaluation': evaluations, 'engineering_smokes': engineering, 'failed_attempts': failed, 'superseded_generation_completed_costs': superseded, 'failed_generation_without_completed_cost_record': failure_logs, 'ssu_mask': {'blocks': 500, 'tokens_per_block': 2048, 'forward_tokens': 1024000, 'process_wall_seconds': mask_seconds, 'mask_sha256': sha(DATA / 'ssu-mask-v1')}, 'sampled_gpu_peak_mib_by_role': dict(peaks), 'sampled_max_single_process_rss_kib_by_role': dict(rss), 'resource_sample_counts': dict(sampled), 'resource_sample_sha256': sample_sources, 'new_disk_usage_bytes': sizes, 'directory_and_sampler_scope': 'Directory sizes and folder-wide process samples include later supplemental/development/rebalanced-data additions; they are not exclusive footprints of the original three trainings. Precise original per-training allocator peaks come from its own accounting file.', 'additional_cost_evidence': {'supplemental': 'results/aya_4b_20261105/supplemental-analysis-v1/cost.json', 'decoding_development': 'results/aya_4b_20261105/translation-decoding-development-analysis-v1/results.json', 'xnli_format_development': 'results/aya_4b_20261105/xnli-format-control-analysis-v1/results.json', 'data_rebalanced_training': 'results/aya_4b_20261105/translation-rebalanced-training-audit-v1.json'}, 'trainer_runtime_sum_seconds': sum((x['trainer_runtime_seconds'] for x in training.values())), 'reported_evaluation_runtime_sum_seconds': sum((x['wall_seconds'] for x in evaluations)), 'limitations': ['Sums are cumulative process/evaluation wall time across parallel GPUs, not project elapsed time or CUDA kernel time', 'v1 sampler missed some HF/vLLM child processes; v2 includes verified own descendants; observed peaks remain lower bounds', 'RSS includes shared pages and allocator caches; it is not unique total system RAM', 'Final-only weight exports have no optimizer checkpoints; smoke exports and failures are retained'], 'created_at': datetime.datetime.now().astimezone().isoformat()}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k in ['status', 'trainer_runtime_sum_seconds', 'reported_evaluation_runtime_sum_seconds', 'sampled_gpu_peak_mib_by_role', 'new_disk_usage_bytes']}, indent=2))
if __name__ == '__main__':
    main()
