"""One finite CPU-only drain for the nine authorized Base outputs; no retries."""
import sys, time, json, collections, subprocess
from pathlib import Path
from common import *
R = ROOT / 'results/math_sft_base_fft'
done = set()
start = time.time()
out = R / 'base-census-polymath-v1'
assert not out.exists()
out.mkdir()
assert (R / 'polymath-calibration-v3/CALIBRATION_DONE.json').exists()
while len(done) < len(LANGS):
    changed = False
    for worker in [1, 2, 3]:
        source = R / f'base-census-worker{worker}-v1/predictions.jsonl'
        if not source.exists():
            continue
        by = collections.defaultdict(list)
        for line in source.open():
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            by[x['lang']].append(x)
        for lang, xs in by.items():
            if lang in done or len(xs) != 1750:
                continue
            assert len({x['id'] for x in xs}) == 1750
            snap = out / (lang + '-input')
            snap.mkdir()
            (snap / 'predictions.jsonl').write_text(''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in xs)))
            dump(snap / 'provenance.json', {'worker_source': str(source), 'rows': 1750, 'source_prompt_protocol_sha256': sha(DATA / 'prepared-v1/protocol.json'), 'snapshot_sha256': sha(snap / 'predictions.jsonl')})
            cmd = [sys.executable, str(Path(__file__).with_name('score_polymath.py')), '--input', str(snap), '--output', str(out / (lang + '-score'))]
            with (out / (lang + '.log')).open('x') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            done.add(lang)
            changed = True
            dump(out / 'status.json', {'completed_languages': sorted(done), 'total': 9, 'seconds': time.time() - start})
            print('LANGUAGE_SCORED', lang, flush=True)
    if not changed:
        time.sleep(15)
dump(out / 'DONE.json', {'status': 'COMPLETE', 'languages': sorted(done), 'seconds': time.time() - start})
