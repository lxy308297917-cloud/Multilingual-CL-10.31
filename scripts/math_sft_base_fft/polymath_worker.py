"""Executed ONLY inside verified filesystem/user/network/PID sandbox."""
import sys, json, hashlib
from kernel_limits import driver_limits
driver_limits()
sys.path.insert(0, '/official/eval')
import bundled.symeval as pkg
from bundled.symeval.core import EvaluatorMath
pkg.EvaluatorMath = EvaluatorMath
from scripts import math_equal
from run_eval import extract_boxed_content
spec = json.load(open('/input.json'))
out = []
for x in spec['rows']:
    boxes = extract_boxed_content(x['output'])
    pred = boxes[0] if boxes else None
    correct = bool(math_equal(pred, str(x['answer'])))
    out.append({'id': x['id'], 'extracted': pred, 'correct': correct, 'box_parse': pred is not None})
print('POLYMATH_RESULT ' + json.dumps(out, ensure_ascii=False), flush=True)
