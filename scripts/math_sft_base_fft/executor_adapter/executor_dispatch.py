"""One-shot CPU evaluation, no retries, no new model generation."""
from profile_paths import resolve
import concurrent.futures as cf
import hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from isolation import A, run
ROOT = Path(resolve('ROOT'))
D = Path(resolve('EXECUTOR'))
RAW = Path(resolve('CODE_DATA') + '/raw')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, v):
    p = Path(p)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(v, indent=2) + '\n')
    tmp.replace(p)

def evaluate(job):
    p = Path(job['directory'])
    p.mkdir(parents=True, exist_ok=False)
    f = p / 'input.json'
    f.write_text(json.dumps({'dataset': job['dataset'], 'problem': job['problem'], 'solution': job['solution']}))
    result = {'method': job['method'], 'dataset': job['dataset'], 'task_id': job['problem']['task_id'], 'input_sha256': sha(f), 'solution_sha256': hashlib.sha256(job['solution'].encode()).hexdigest()}
    r = run(D / 'evaluate_one.py', f, p / 'stdout.log', p / 'stderr.log', wall=120)
    result['execution'] = r
    out = (p / 'stdout.log').read_text(errors='replace')
    result['stderr_sha256'] = sha(p / 'stderr.log')
    result['stdout_sha256'] = sha(p / 'stdout.log')
    try:
        if r['exit_code'] != 0 or r['wall_timeout']:
            raise ValueError('outer sandbox exited or timed out')
        line = out.splitlines()[-1]
        if not line.startswith('EVALPLUS_RESULT '):
            raise ValueError('trusted final result missing')
        score = json.loads(line[len('EVALPLUS_RESULT '):])
        for k in ['task_id', 'dataset', 'solution_sha256']:
            if score[k] != result[k]:
                raise ValueError('result identity mismatch ' + k)
        if not all((score[k] in ['pass', 'fail', 'timeout'] for k in ['base', 'plus'])):
            raise ValueError('invalid official status')
        if score['base_pass'] != (score['base'] == 'pass') or score['plus_pass'] != (score['base'] == 'pass' and score['plus'] == 'pass'):
            raise ValueError('pass definition mismatch')
        result.update(score)
        result['infrastructure_status'] = 'COMPLETE'
    except Exception as e:
        result['infrastructure_status'] = 'FAILED'
        result['error'] = str(e)
    dump(p / 'result.json', result)
    return result

def phase(jobs, out, name):
    results = []
    failure = None
    start = time.time()
    status = out / 'status.json'
    queue = iter(jobs)
    with cf.ProcessPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(evaluate, next(queue)): None for _ in range(min(2, len(jobs)))}
        while pending:
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in done:
                pending.pop(f)
                r = f.result()
                results.append(r)
                if r['infrastructure_status'] != 'COMPLETE':
                    failure = r
                if name == 'canonical_calibration' and (not (r.get('base_pass') and r.get('plus_pass'))):
                    failure = r
                dump(status, {'phase': name, 'completed': len(results), 'total': len(jobs), 'failed': bool(failure), 'updated_at': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.time() - start})
                if not failure:
                    job = next(queue, None)
                    if job is not None:
                        pending[pool.submit(evaluate, job)] = None
            if failure:
                for f in pending:
                    f.cancel()
                break
    dump(out / (name + '-results.json'), results)
    if failure:
        dump(out / 'FAILED.json', {'phase': name, 'failure': failure})
        raise RuntimeError('Stopped after failed gate or infrastructure issue; inspect FAILED.json')
    return results
