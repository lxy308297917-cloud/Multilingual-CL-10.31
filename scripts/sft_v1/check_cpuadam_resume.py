"""CPU-only regression for converted optimizer/scheduler references at resume.

Loads the existing CPUAdam binary read-only so this diagnostic cannot rebuild
an extension used by live training. Optimizer updates themselves are unchanged.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from transformers import get_cosine_schedule_with_warmup
from accelerate.utils.deepspeed import map_pytorch_optim_to_deepspeed
from deepspeed.ops.adam import cpu_adam


def run():
    library = Path('/path/to/storage/torch-extensions-qwen3/cpu_adam/cpu_adam.so')
    spec = importlib.util.spec_from_file_location('cpu_adam', library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cpu_adam.CPUAdamBuilder.load = lambda self: module

    def setup(values):
        p = torch.nn.Parameter(values.clone())
        original = torch.optim.AdamW([p], lr=.01, weight_decay=0)
        sched = get_cosine_schedule_with_warmup(original, 1, 8)
        actual = map_pytorch_optim_to_deepspeed(original)
        return p, actual, sched

    def steps(p, actual, sched, count):
        trace = []
        for _ in range(count):
            p.grad = torch.tensor([.13, -.27])
            trace.append({'actual_lr': actual.param_groups[0]['lr'],
                          'scheduler_reported_lr': sched.get_last_lr()[0]})
            actual.step(); sched.step()
        return trace

    p, actual, sched = setup(torch.tensor([.8, 1.2]))
    prefix = steps(p, actual, sched, 4)
    saved = copy.deepcopy(actual.state_dict())
    schedule = copy.deepcopy(sched.state_dict())
    values = p.detach().clone()
    tail = steps(p, actual, sched, 4)
    expected = p.detach().clone()
    report = {}
    for fixed in (False, True):
        q, opt, scheduler = setup(values)
        opt.load_state_dict(copy.deepcopy(saved))
        scheduler.load_state_dict(copy.deepcopy(schedule))
        linked = opt.param_groups[0] is scheduler.optimizer.param_groups[0]
        if fixed:
            scheduler.optimizer = opt
        trace = steps(q, opt, scheduler, 4)
        report['rebound' if fixed else 'unfixed'] = {
            'group_reference_same_before_fix': linked, 'trace': trace,
            'matches_continuous': torch.equal(q.detach(), expected),
            'max_abs_difference': float((q.detach()-expected).abs().max())}
    assert not report['unfixed']['matches_continuous']
    assert report['rebound']['matches_continuous']
    return {'status': 'confirmed CPU-only optimizer-reference regression and repair',
            'tests': report, 'continuous_prefix': prefix, 'continuous_tail': tail,
            'extension_path': str(library),
            'extension_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
            'scope': 'Installed Accelerate conversion and DeepSpeedCPUAdam with tiny CPU vectors; corrected real1.7B resume still needs its GPU check. Current formal SSU started from scratch.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    result = run()
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['tests'], indent=2))
