"""All fixed GPU benchmark phases for one epoch, preserving exact inputs."""
from profile_paths import resolve
import argparse, sys, subprocess, json
from pathlib import Path
from common import *
ap = argparse.ArgumentParser()
ap.add_argument('--model', type=Path, required=True)
ap.add_argument('--lang', required=True)
ap.add_argument('--role', required=True)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()
assert not a.output.exists()
a.output.mkdir(parents=True)
assert (a.model / 'VALIDATED_MATH_SFT.json').exists()
assert json.loads((a.model / 'VALIDATED_MATH_SFT.json').read_text())['every_source_example_covered']
script = Path(__file__).parent
evalpy = resolve('EVAL_PY')
subprocess.run([evalpy, str(script / 'evaluate_math.py'), '--langs', a.lang, '--model', str(a.model), '--role', a.role, '--output', str(a.output / 'math')], check=True)
subprocess.run([sys.executable, str(script / 'evaluate_general.py'), '--model', str(a.model), '--role', a.role, '--output', str(a.output / 'general'), '--tasks', 'core', 'mmlu', 'code'], check=True)
dump(a.output / 'GPU_PHASE_DONE.json', {'status': 'COMPLETE', 'role': a.role, 'model': str(a.model), 'language': a.lang, 'polymath_equivalence_and_code_execution': 'PENDING_CPU_PHASE'})
