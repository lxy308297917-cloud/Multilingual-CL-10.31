"""Measured walls and actual exposures; never equates these with CUDA kernel time."""
from pathlib import Path
import os
import argparse
import collections
import csv
import datetime
import hashlib
import json

R = Path(__file__).resolve().parents[3]
K = Path(os.environ['P2_CHECKPOINT_ROOT'])
L = Path(os.environ['P2_LOG_ROOT'])
D = Path(os.environ['OLD_POOL_DIR'])
O = R / 'research/functional_representation_20261109/p2'

def load(p):
    return json.loads(p.read_text())

def lines(p):
    return [json.loads(x) for x in p.open()]

def write_csv(p, rows):
    if not rows:return
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run_cost(path, label, role):
    result = load(path / ('TRAINING_DONE.json' if role == 'formal' else 'SMOKE_DONE.json'))
    accounting = load(path / 'accounting-rank0.json')
    initial = accounting['resume_initial_step']
    trace = [x for x in lines(path / 'exposure-rank0.jsonl') if x['update_step'] > initial]
    old = [x for x in lines(path / 'old_exposure.jsonl') if x['step'] > initial]
    setting = load(path / 'old_accounting.json')
    pool = {x['id']: x for x in lines(D / 'build_pool.jsonl')}
    mb = setting['selected_target_microbatch']
    mode = setting['mode']
    assert len(trace) == (result['steps'] - initial) * 16
    assert len(old) == (0 if mode == 'all_layer' else (result['steps'] - initial) * setting['gradient_accumulation_steps'])
    return dict(run=label, role=role, mode=mode, optimizer_initial_step=initial, optimizer_final_step=result['steps'],
                target_samples_this_process=len(trace), target_input_tokens_this_process=sum(x['input_tokens'] for x in trace),
                target_ce_supervised_tokens_this_process=sum(x['shifted_assistant_tokens'] for x in trace),
                target_microbatch_forward_backward_calls=len(trace) // mb,
                old_samples_this_process=len(old), old_input_tokens_this_process=sum(x['input_tokens'] for x in old),
                old_reference_answer_tokens_this_process=sum(x['assistant_tokens'] for x in old),
                old_ce_supervised_tokens_this_process=sum(x['assistant_tokens'] for x in old) if mode == 'ce_replay' else 0,
                feature_answer_positions_this_process=sum(len(pool[x['id']]['answer_prediction_slots']) for x in old) if mode == 'full_feature' else 0,
                feature_layer_positions_this_process=36 * sum(len(pool[x['id']]['answer_prediction_slots']) for x in old) if mode == 'full_feature' else 0,
                old_forward_backward_calls=len(old),
                elapsed_accounting_process_seconds=result['elapsed_seconds'],
                peak_cuda_allocated_bytes=result['peak_allocated_bytes'],
                peak_cuda_reserved_bytes=accounting['max_memory_reserved'],
                target_microbatch=mb, gradient_accumulation=setting['gradient_accumulation_steps'])

def main():
    ap = argparse.ArgumentParser();ap.add_argument('--partial', action='store_true');args = ap.parse_args()
    training, engineering, endpoints = [], [], []
    for mode in ['all_layer', 'full_feature', 'ce_replay']:
        path = K / (mode + '_final_v1')
        if (path / 'TRAINING_DONE.json').exists():training.append(run_cost(path, mode, 'formal'))
        for stage in ['reading', 'core', 'summary', 'gmmlu']:
            job = L / 'evaluation' / mode / (stage + '.job')
            if not (job / 'DONE.json').exists():continue
            done = load(job / 'DONE.json')
            endpoints.append(dict(mode=mode, stage=stage,
                    wall_seconds_including_queue_model_load_gpu_and_cpu_scoring=done['elapsed_seconds_including_queue'],
                    completed=True))
    for path in sorted(K.glob('engineering_*')):
        if (path / 'SMOKE_DONE.json').exists():engineering.append(run_cost(path, path.name, 'engineering'))
    complete = len(training) == 3 and len(endpoints) == 12
    if not args.partial:assert complete
    cache = D / 'build_reference_features.safetensors'
    report = dict(complete=complete, formal_training_jobs_complete=len(training), formal_evaluation_stage_jobs_complete=len(endpoints),
                  engineering_gpu_jobs=len(engineering), maximum_account_gpus=4, training_seeds=1,
                  formal_training_reported_process_hours=sum(x['elapsed_accounting_process_seconds'] for x in training) / 3600,
                  formal_endpoint_stage_wall_hours_including_wait=sum(x['wall_seconds_including_queue_model_load_gpu_and_cpu_scoring'] for x in endpoints) / 3600,
                  engineering_process_seconds=sum(x['elapsed_accounting_process_seconds'] for x in engineering),
                  engineering_actual_target_samples=sum(x['target_samples_this_process'] for x in engineering),
                  engineering_actual_old_samples=sum(x['old_samples_this_process'] for x in engineering),
                  reference_cache=dict(bytes=cache.stat().st_size, sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                                       build_examples=128, reused_from_P1=True, standalone_build_cost='NOT_MEASURED',
                                       previous_P1_generation_and_gradient_diagnostic_costs_reported_separately=True),
                  normalization_reference=dict(historical_unprotected_full_lora_used=True,
                      information='Per-capability build drift from previously trained400-step unprotected Full LoRA',
                      legacy_training_seconds=(datetime.datetime.fromisoformat(load(Path(os.environ['LEGACY_FULL_ADAPTER']) / 'TRAINING_DONE.json')['ended_at']) - datetime.datetime.fromisoformat(load(Path(os.environ['LEGACY_FULL_ADAPTER']) / 'TRAINING_DONE.json')['started_at'])).total_seconds(),
                      actual_P2_new_training_cost=False, standalone_method_cost_not_free=True),
                  interpretation=['Process walls include loading and CPU work; not CUDA kernel time',
                     'Stage wall sums overlap and include queue waits; they are not total elapsed wall time',
                     'Old forward timer includes synchronization and can include queued target work; not pure incremental GPU time',
                     'HF total_flos is a target-only estimator and is not used as combined measured compute',
                     'Target gradient-checkpoint recomputations are not included in top-level forward/backward call counts',
                     'Full-feature uses reference answer text but no old CE labels; CE replay explicitly uses old CE labels',
                     'Same old samples and calls do not guarantee equal FLOPs between feature and CE objectives',
                     'GPU final tensors not bitwise repeatable; failed initial bitwise recovery assertion and repeated control retained'])
    write_csv(O / 'training_costs.csv', training);write_csv(O / 'engineering_costs.csv', engineering);write_csv(O / 'evaluation_costs.csv', endpoints)
    (O / 'cost_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['interpretation', 'reference_cache']}))

if __name__ == '__main__':
    main()
