"""Real EvalPlus pass@1 through the already verified isolated executor only."""
from profile_paths import resolve
import sys, os, json, argparse, time, hashlib, subprocess
from pathlib import Path
from common import *
APP = Path(resolve('SANDBOX_WORKSPACE'))
EXEC = Path(resolve('EXECUTOR'))
sys.path.insert(0, str(EXEC))
import executor_dispatch as executor
from fixed_namespace_policy import require_fixed_namespace_proof

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generation', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--role', required=True)
    ap.add_argument('--calibration', type=Path)
    a = ap.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    manifest = json.loads((a.generation / 'manifest.json').read_text())
    assert manifest['scope'] == 'full' and manifest['benchmark_id'] == 'humaneval_plus_v2' and (len(manifest['sample_ids']) == 164)
    assert (a.generation / 'DONE.json').exists()
    raw = Path(resolve('CODE_DATA') + '/raw/HumanEvalPlus-v0.1.10.jsonl')
    assert sha(raw) == manifest['benchmark_protocol']['source']['archive_uncompressed_sha256']
    problems = {x['task_id']: x for x in rows(raw)}
    assert len(problems) == 164
    proof = json.loads((EXEC / 'execution-proof.json').read_text())
    require_fixed_namespace_proof(proof)
    assert proof['resource_enforcement_test_passed']
    paths = [EXEC / 'isolation.py', EXEC / 'kernel_limits.py', EXEC / 'evaluate_one.py', EXEC / 'runtime-manifest.json', EXEC / 'execution-proof.json', EXEC / 'fixed_namespace_policy.py']
    identity = {p.name: sha(p) for p in paths}
    identity['raw_sha256'] = sha(raw)
    codepy = resolve('CODE_PY')
    with (a.output / 'sanitize.log').open('x') as log:
        subprocess.run([codepy, str(ROOT / 'scripts/bench_v2/analyze_code.py'), '--run', str(a.generation), '--official-data', str(raw), '--output', str(a.output / 'static')], stdout=log, stderr=subprocess.STDOUT, check=True)
    canon = [{'method': 'canonical', 'dataset': 'humaneval', 'problem': p, 'solution': p['prompt'] + p['canonical_solution'], 'directory': str(a.output / 'canonical' / p['task_id'].split('/')[-1])} for p in problems.values()]
    if a.calibration:
        prior = json.loads((a.calibration / 'executor_identity.json').read_text())
        assert prior == identity
        cr = json.loads((a.calibration / 'canonical_calibration-results.json').read_text())
        assert len(cr) == 164 and all((x.get('base_pass') and x.get('plus_pass') and (x['infrastructure_status'] == 'COMPLETE') for x in cr))
        dump(a.output / 'calibration_reuse.json', {'source': str(a.calibration), 'identity': identity, 'reference_tasks': 164, 'passed': 164})
    else:
        executor.phase(canon, a.output, 'canonical_calibration')
    dump(a.output / 'executor_identity.json', identity)
    samples = rows(a.output / 'static/sanitized_samples.jsonl')
    assert len(samples) == 164 and {x['task_id'] for x in samples} == set(problems)
    jobs = [{'method': a.role, 'dataset': 'humaneval', 'problem': problems[x['task_id']], 'solution': x['solution'], 'directory': str(a.output / 'samples' / x['task_id'].split('/')[-1])} for x in samples]
    start = time.monotonic()
    results = executor.phase(jobs, a.output, 'model_evaluation')
    assert len(results) == 164
    summary = {'status': 'COMPLETE', 'n': 164, 'pass_at_1_base': sum((x['base_pass'] for x in results)) / 164, 'pass_at_1_plus': sum((x['plus_pass'] for x in results)) / 164, 'base_correct': sum((x['base_pass'] for x in results)), 'plus_correct': sum((x['plus_pass'] for x in results)), 'seconds_model_execution': time.monotonic() - start, 'protocol': 'EvalPlus0.3.1+find_zero-bookkeeping; 4GiB/process,2 processes,2 parallel isolated namespaces,120s outer wall; real execution; plus requires base and plus pass', 'generation_manifest_sha256': sha(a.generation / 'manifest.json'), 'generation_predictions_sha256': sha(a.generation / 'predictions.jsonl'), 'executor_identity': identity}
    dump(a.output / 'scores.json', summary)
    dump(a.output / 'DONE.json', summary)
    print(json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
