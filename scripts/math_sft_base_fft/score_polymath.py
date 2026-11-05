"""Official PolyMath equivalence in the existing hardened namespace sandbox."""
from profile_paths import resolve
import argparse, sys, collections, json, concurrent.futures, time, resource
from pathlib import Path
from common import *
APP = Path(resolve('SANDBOX_WORKSPACE'))
sys.path.insert(0, resolve('EXECUTOR'))
import isolation
oldcommand = isolation.command

def command(script, input_path=None):
    cmd = oldcommand(script, input_path)
    old = resolve('CODE_ENV') + '/lib/python3.11/site-packages'
    new = resolve('POLY_ENV') + '/lib/python3.11/site-packages'
    cmd = [new if x == old else x for x in cmd]
    cmd[cmd.index('/kernel_limits.py') - 1] = str(Path(__file__).with_name('polymath_kernel_limits.py'))
    idx = cmd.index('--remount-ro', cmd.index('--ro-bind', cmd.index('/entry.py') + 1)) if False else cmd.index('--remount-ro', cmd.index('/kernel_limits.py') + 1)
    cmd[idx:idx] = ['--ro-bind', str(DATA / 'sources/PolyMath-official'), '/official']
    return cmd
isolation.command = command
oldlimits = isolation.outer_limits

def outer_limits():
    oldlimits()
    resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 ** 2, 8 * 1024 ** 2))
isolation.outer_limits = outer_limits

def one(job):
    folder, xs = job
    folder.mkdir(parents=True)
    inp = folder / 'input.json'
    dump(inp, {'rows': xs})
    res = isolation.run(Path(__file__).with_name('polymath_worker.py'), inp, folder / 'stdout.log', folder / 'stderr.log', wall=120)
    dump(folder / 'execution.json', res)
    assert res['exit_code'] == 0 and (not res['wall_timeout']), (folder, res)
    lines = (folder / 'stdout.log').read_text().splitlines()
    assert lines[-1].startswith('POLYMATH_RESULT '), (folder, lines[-3:])
    ys = json.loads(lines[-1][len('POLYMATH_RESULT '):])
    assert [x['id'] for x in xs] == [y['id'] for y in ys]
    return ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--calibrate', action='store_true')
    a = ap.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    if a.calibrate:
        cases = [('fraction', 'Answer: \\\\boxed{\\\\frac{1}{2}}', '0.5', True), ('symbolic', '\\\\boxed{(x+1)^2}', 'x^2+2*x+1', True), ('wrong', '\\\\boxed{3}', '4', False), ('first_box', '\\\\boxed{2} then \\\\boxed{3}', '2', True), ('no_box', 'answer 2', '2', False), ('sqrt', '\\\\boxed{\\\\sqrt{4}}', '2', True)]
        xs = [dict(id=i, output=o.replace('\\\\', '\\'), answer=ans, expected=e) for i, o, ans, e in cases]
        ys = one((a.output / 'sandbox', xs))
        assert [x['correct'] for x in ys] == [x['expected'] for x in xs], ys
        dump(a.output / 'CALIBRATION_DONE.json', {'cases': ys, 'official_code_revision': 'fbf4e41cae78687d6be7447dbea897357c06aaa7', 'worker_sha256': sha(Path(__file__).with_name('polymath_worker.py'))})
        print(ys)
        return
    xs = [r for r in rows(a.input / 'predictions.jsonl') if r['benchmark'] == 'polymath']
    assert xs
    jobs = [(a.output / f'chunk-{i:04d}', xs[i:i + 24]) for i in range(0, len(xs), 24)]
    ys = []
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        for v in ex.map(one, jobs):
            ys += v
            print(json.dumps({'scored': len(ys), 'total': len(xs), 'seconds': time.monotonic() - start}), flush=True)
    maps = {x['id'] + ':' + x['lang']: x for x in xs}
    groups = collections.defaultdict(list)
    scored = []
    for x, y in zip(xs, ys):
        assert x['id'] == y['id']
        z = dict(y, lang=x['lang'], level=x['level'])
        scored.append(z)
        groups[x['lang'] + ':' + x['level']].append(z)
    scores = {k: {'samples': len(v), 'correct': sum((x['correct'] for x in v)), 'accuracy': sum((x['correct'] for x in v)) / len(v), 'box_parse_rate': sum((x['box_parse'] for x in v)) / len(v)} for k, v in groups.items()}
    for lang in sorted({x['lang'] for x in xs}):
        vals = [scores[lang + ':' + l]['accuracy'] for l in ['low', 'medium', 'high', 'top']]
        scores[lang + ':DW-ACC'] = {'official_percent': round(sum((2 ** i * round(v * 100, 1) for i, v in enumerate(vals))) / 15, 1), 'unrounded_accuracy': sum((2 ** i * v for i, v in enumerate(vals))) / 15}
    (a.output / 'scored.jsonl').write_text(''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in scored)))
    dump(a.output / 'scores.json', scores)
    dump(a.output / 'DONE.json', {'samples': len(xs), 'seconds': time.monotonic() - start, 'input_predictions_sha256': sha(a.input / 'predictions.jsonl'), 'official_code_revision': 'fbf4e41cae78687d6be7447dbea897357c06aaa7', 'code_sha256': {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('polymath_worker.py')]}})
if __name__ == '__main__':
    main()
