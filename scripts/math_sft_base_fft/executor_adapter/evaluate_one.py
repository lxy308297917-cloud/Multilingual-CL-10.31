"""Run one immutable EvalPlus problem and candidate inside an OS sandbox."""
import json, hashlib, sys
from kernel_limits import driver_limits, child_limits
driver_limits()
import evalplus.eval as ev
import evalplus.evaluate as evaluator
import inspect
source = inspect.getsource(ev.unsafe_execute)
old = 'assert abs(_poly(*inp, out)) <= atol\n                            continue'
new = 'assert abs(_poly(*inp, out)) <= atol\n                            details[i] = True\n                            progress.value += 1\n                            continue'
if source.count(old) != 1:
    raise RuntimeError('Unexpected upstream find_zero oracle source')
exec(compile(source.replace(old, new), '<evalplus-0.3.1-find-zero-bookkeeping-fix>', 'exec'), ev.__dict__)
from evalplus.data.mbpp import mbpp_deserialize_inputs
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
spec = json.load(open('/input.json'))
problem = spec['problem']
dataset = spec['dataset']
if dataset == 'mbpp':
    for key in ['base_input', 'plus_input']:
        problem[key] = mbpp_deserialize_inputs(problem['task_id'], problem[key])
evaluator.CACHE_DIR = '/work/oracles'
from evalplus.gen.util import trusted_exec
expected = {}
for split in ['base', 'plus']:
    expected[split], expected[split + '_time'] = trusted_exec(problem['prompt'] + problem['canonical_solution'], problem[split + '_input'], problem['entry_point'], record_time=True, output_not_none=dataset == 'mbpp' and problem['entry_point'] in MBPP_OUTPUT_NOT_NONE_TASKS)
original_guard = ev.reliability_guard

def guarded(*args, **kwargs):
    original_guard(*args, **kwargs)
    child_limits()
ev.reliability_guard = guarded
result = evaluator.check_correctness(dataset, 0, problem, spec['solution'], expected, fast_check=True, min_time_limit=1.0, gt_time_limit_factor=4.0)
summary = {'task_id': problem['task_id'], 'dataset': dataset, 'solution_sha256': hashlib.sha256(spec['solution'].encode()).hexdigest(), 'base': result['base'][0], 'plus': result['plus'][0], 'base_pass': result['base'][0] == ev.PASS, 'plus_pass': result['base'][0] == ev.PASS and result['plus'][0] == ev.PASS, 'scorer_patch': 'evalplus-0.3.1-find_zero-success-bookkeeping', 'base_tests': len(problem['base_input']), 'plus_tests': len(problem['plus_input'])}
print('EVALPLUS_RESULT ' + json.dumps(summary), flush=True)
