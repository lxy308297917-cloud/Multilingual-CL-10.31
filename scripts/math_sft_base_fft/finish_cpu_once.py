"""Finite CPU scoring for shared Base and all six predeclared route-A epochs."""
import sys, time, json, subprocess
from pathlib import Path
from common import *
R = ROOT / 'results/math_sft_base_fft'
P = Path(__file__).parent
done = set()
out = R / 'cpu-drain-routeA-v1'
assert not out.exists()
out.mkdir()
start = time.time()

def launch(name, cmd):
    with (out / (name + '.log')).open('x') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
while True:
    roles = [('base', R / 'base-general-v1', None)]
    select = R / 'base-language-screening-v1/selection.json'
    if select.exists():
        selection = json.loads(select.read_text())
        assert selection['status'] == 'COMPLETE'
        roles += [(f'A-{lang}-E{ep}', R / 'A' / lang / f'epoch{ep}' / 'general', R / 'A' / lang / f'epoch{ep}' / 'math') for lang in selection['selected_languages'] for ep in [1, 2]]
    changed = False
    for name, general, math in roles:
        if math and (math / 'GENERATION_DONE.json').exists() and (name + '-poly' not in done):
            launch(name + '-poly', [sys.executable, str(P / 'score_polymath.py'), '--input', str(math), '--output', str(math.parent / 'polymath')])
            done.add(name + '-poly')
            changed = True
        generation = general / 'code/generation'
        calibration = R / 'base-general-v1/code-execution'
        if (generation / 'DONE.json').exists() and name + '-code' not in done and (name == 'base' or (calibration / 'DONE.json').exists()):
            cmd = [sys.executable, str(P / 'execute_code.py'), '--generation', str(generation), '--output', str(general / 'code-execution'), '--role', name]
            if name != 'base':
                cmd += ['--calibration', str(calibration)]
            launch(name + '-code', cmd)
            done.add(name + '-code')
            changed = True
    dump(out / 'status.json', {'completed': sorted(done), 'expected': 13, 'seconds': time.time() - start})
    if len(done) == 13:
        dump(out / 'DONE.json', {'status': 'COMPLETE', 'completed': sorted(done), 'seconds': time.time() - start})
        break
    if not changed:
        time.sleep(20)
