"""Frozen general benchmarks, no score-based checkpoint selection."""
from profile_paths import resolve
import argparse, os, sys, json, subprocess, time
from pathlib import Path
from common import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, default=MODEL)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tasks', nargs='+', choices=['core', 'mmlu', 'code'], default=['core', 'mmlu', 'code'])
    ap.add_argument('--role', default='base')
    a = ap.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    assert gpu.startswith('GPU-') and ',' not in gpu
    trainpy = resolve('TRAIN_PY')
    evalpy = resolve('EVAL_PY')
    weights = {p.name: sha(p) for p in sorted(a.model.glob('*.safetensors'))}
    assert weights
    if a.model != MODEL:
        check = json.loads((a.model / 'EPOCH_CHECKPOINT.json').read_text())
        assert check['weights_sha256'] == weights
    env = dict(os.environ, VLLM_USE_V1='0', VLLM_WORKER_MULTIPROC_METHOD='spawn', NLTK_DATA=resolve('NLTK'))
    hf = f'pretrained={a.model},tokenizer={MODEL},dtype=bfloat16,enable_thinking=False'
    vv = f'pretrained={a.model},tokenizer={MODEL},dtype=bfloat16,max_model_len=8192,gpu_memory_utilization=0.85,max_num_seqs=32,max_num_batched_tokens=8192,enable_prefix_caching=True,seed=42,enforce_eager=True,enable_thinking=False'
    commands = {'core': [evalpy, str(ROOT / 'scripts/sft_v1/harness.py'), 'run', '--model', 'vllm', '--model_args', vv, '--tasks', 'sft_v1_gsm8k,sft_v1_ifeval', '--include_path', str(ROOT / 'configs/sft_v1/eval_tasks'), '--apply_chat_template', '--batch_size', 'auto', '--output_path', str(a.output / 'core/results'), '--log_samples', '--seed', '42'], 'mmlu': [str(Path(trainpy).parent / 'lm-eval'), 'run', '--model', 'hf', '--model_args', hf, '--tasks', 'mmlu_locked', '--include_path', resolve('MMLU_TASKS'), '--num_fewshot', '5', '--apply_chat_template', '--batch_size', '4', '--device', 'cuda:0', '--output_path', str(a.output / 'mmlu/results'), '--log_samples', '--seed', '42'], 'code': [evalpy, str(ROOT / 'scripts/bench_v2/generate.py'), '--dataset', resolve('CODE_DATA') + '/prepared-v1/humaneval_plus_v2', '--model', str(a.model), '--tokenizer', str(MODEL), '--method', 'base' if a.role == 'base' else 'full', '--gpu-uuid', gpu, '--output', str(a.output / 'code/generation'), '--scope', 'full', '--batch-size', '32', '--max-user-gpus', '4']}
    taskfiles = [p for p in (ROOT / 'configs/sft_v1/eval_tasks').rglob('*') if p.is_file() and p.suffix in ['.py', '.yaml']] + [p for p in Path(resolve('MMLU_TASKS')).rglob('*.yaml')]
    dump(a.output / 'launch.json', {'role': a.role, 'weights_sha256': weights, 'commands': {k: commands[k] for k in a.tasks}, 'frozen_task_files': {str(p): sha(p) for p in taskfiles}, 'code_manifest_sha256': sha(resolve('CODE_DATA') + '/prepared-v1/humaneval_plus_v2/manifest.json'), 'source_sha256': sha(__file__), 'formal_test_selection': False})
    scores = {}
    start = time.monotonic()
    for task in a.tasks:
        folder = a.output / task
        folder.mkdir()
        extra = {'SFT_VLLM_TRACE': str(folder / 'completion-metadata.jsonl')} if task == 'core' else {}
        with (folder / 'run.log').open('x') as log:
            subprocess.run(commands[task], cwd=ROOT, env=dict(env, **extra), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)
        if task != 'code':
            fs = list(folder.glob('results/**/results_*.json'))
            assert len(fs) == 1
            scores[task] = json.loads(fs[0].read_text())['results']
        else:
            scores[task] = {'status': 'GENERATION_COMPLETE_EXECUTION_PENDING', 'samples': 164}
        dump(folder / 'TASK_DONE.json', {'task': task, 'status': 'COMPLETE' if task != 'code' else 'GENERATION_COMPLETE_EXECUTION_PENDING', 'seconds_since_start': time.monotonic() - start})
        print(json.dumps({'task': task, 'seconds': time.monotonic() - start}), flush=True)
    dump(a.output / 'scores.json', scores)
    dump(a.output / 'GPU_PHASE_DONE.json', {'status': 'COMPLETE', 'tasks': a.tasks, 'seconds': time.monotonic() - start, 'weights_sha256': weights})
if __name__ == '__main__':
    main()
