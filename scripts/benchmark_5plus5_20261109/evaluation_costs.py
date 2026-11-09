"""CPU-only observed FA/FFT evaluation costs; never starts or restarts jobs."""
from pathlib import Path
import datetime
import hashlib
import json

R = Path('/path/to/project')
L = Path('/path/to/logs/qwen3-multilingual')
FAMILIES = {
    'function_aware_formal_20261109': 'function-aware-formal-20261109',
    'fft_representation_20261109': 'fft-representation-20261109',
}


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(folder, status, terminal):
    command = folder / 'inference.log.command.json'
    result = dict(model=folder.parent.name, stage=folder.name,
                  output=str(folder), status=status,
                  terminal_record_sha256=sha(terminal))
    if not command.exists():
        result.update(postlaunch_wall_seconds=None,
                      missing_reason='No launch timestamp: unknown, not zero.')
        return result
    start = datetime.datetime.fromisoformat(load(command)['started_at']).timestamp()
    end = terminal.stat().st_mtime if status == 'COMPLETE' else datetime.datetime.fromisoformat(load(terminal)['updated_at']).timestamp()
    assert end >= start, (folder, start, end)
    result.update(command_record_sha256=sha(command),
                  postlaunch_wall_seconds=end - start,
                  interval_start_unix_seconds=start,
                  interval_end_unix_seconds=end,
                  end_source='DONE file mtime' if status == 'COMPLETE' else 'terminal state updated_at')
    generation = folder / 'generation/cost.json'
    if generation.exists():
        data = load(generation)
        for key in ['elapsed_seconds', 'input_tokens', 'generated_tokens', 'gpu_peak_allocated_bytes']:
            if key in data:
                result['generation_' + key] = data[key]
        result['generation_cost_sha256'] = sha(generation)
    execution = folder / 'execution/scores.json'
    if execution.exists():
        data = load(execution)
        assert data['status'] == 'COMPLETE'
        result.update(isolated_code_execution_wall_seconds=data['seconds_model_execution'],
                      execution_score_sha256=sha(execution))
    return result


def main():
    for research_family, log_family in FAMILIES.items():
        root = L / log_family
        completed, failed, reused, seen = [], [], [], set()
        for done in sorted((root / 'evaluation').glob('*/*/DONE.json')):
            identity = done.parent.resolve()
            assert identity not in seen, ('Duplicate resolved evaluation', identity)
            seen.add(identity)
            completed.append(record(done.parent, 'COMPLETE', done))
        for terminal in sorted((root / 'evaluation-failed').glob('*/*/state.json')):
            if load(terminal).get('status') == 'FAILED':
                failed.append(record(terminal.parent, 'FAILED_ATTEMPT', terminal))
        for marker in sorted((root / 'evaluation').glob('*/*/REUSED_RESULT.json')):
            data = load(marker)
            reused.append(dict(model=marker.parent.parent.name, stage=marker.parent.name,
                               status=data['status'], marker_sha256=sha(marker),
                               historical_compute_wall_seconds=None,
                               note='Historical inference reused; cancelled waiting duplicate had no inference child. CPU queue/preparation cost not measured; never invent zero.'))
        state = load(root / 'evaluation-schedule.json')
        result = dict(observed_at=datetime.datetime.now().astimezone().isoformat(),
                      status='OBSERVED_COMPLETED_EVALUATIONS_ONLY',
                      scope='This FA/FFT track only; supplement and XQuAD costs are separate. Not a completed full-study cost comparison.',
                      completed_records=completed, failed_attempt_records=failed,
                      reused_records=reused, completed_stage_count=len(completed),
                      completed_postlaunch_wall_sum_hours=sum(x['postlaunch_wall_seconds'] for x in completed if x['postlaunch_wall_seconds'] is not None) / 3600,
                      missing_completed_timing=sum(x['postlaunch_wall_seconds'] is None for x in completed),
                      queued_stages=len(state.get('pending', [])),
                      active_stages=len(state.get('running', state.get('active', []))),
                      caveats=[
                          'Per-stage postlaunch wall includes model load, CPU scoring and synchronization; excludes GPU queue wait. It is not CUDA kernel time.',
                          'Parallel stage wall times are summed; the sum is not calendar elapsed duration.',
                          'Generation and isolated execution timings are nested inside stage totals and must not be added again.',
                          'End timestamps for completed stages are original DONE file mtimes; archival copies are not used as timing sources.',
                          'A failed handoff may not have performed model inference; its wall interval is not labelled GPU computation.',
                          'Unknown historical, incomplete or unavailable costs are not zero.',
                      ], new_gpu_jobs=0, new_training_jobs=0)
        out = R / 'research' / research_family / 'evaluation_costs.json'
        out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(family=research_family, completed=len(completed),
                              failed_attempts=len(failed), reused=len(reused),
                              unknown_completed_timing=result['missing_completed_timing'],
                              postlaunch_wall_sum_hours=result['completed_postlaunch_wall_sum_hours'])))


if __name__ == '__main__':
    main()
