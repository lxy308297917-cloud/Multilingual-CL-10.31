"""Validate an isolated finished training export before formal evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LOG_ROOT = Path('/path/to/workspace/logs/qwen3-multilingual')


def digest(path: Path) -> str:
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            state.update(chunk)
    return state.hexdigest()


def check(model_dir: Path, expected_step: int, log_root: Path, base_model: Path | None = None,
          checkpoint_step: int | None = None, allow_running_checkpoint: bool = False) -> dict:
    model_dir = model_dir.resolve(strict=True)
    if allow_running_checkpoint and checkpoint_step is None:
        raise ValueError('Running-artifact mode requires an explicit checkpoint step')
    if checkpoint_step is not None:
        if model_dir.name != f'checkpoint-{checkpoint_step}':
            raise ValueError('Checkpoint directory and declared step do not match')
        run_id = model_dir.parent.name
    else:
        run_id = model_dir.name
    log_dir = log_root / run_id
    exit_path = log_dir / 'exit_code.txt'
    exit_code = exit_path.read_text().strip() if exit_path.exists() else None
    if not allow_running_checkpoint and exit_code != '0':
        raise ValueError(f'{run_id}: training did not exit successfully')
    if allow_running_checkpoint and not (log_dir / 'run-start.txt').is_file():
        raise ValueError('Missing parent training origin record')
    state = json.loads((model_dir / 'trainer_state.json').read_text())
    artifact_step = checkpoint_step if checkpoint_step is not None else expected_step
    if state['global_step'] != artifact_step or state['max_steps'] != expected_step:
        raise ValueError(f'{run_id}: expected step {artifact_step} of {expected_step}, '
                         f'got {state["global_step"]} of {state["max_steps"]}')
    if checkpoint_step is not None:
        # Locked upstream Trainer saves optimizer/RNG before trainer_state.json.
        if not list(model_dir.glob('rng_state*')):
            raise ValueError('Checkpoint RNG save is incomplete')
        if not ((model_dir / 'optimizer.pt').is_file() or list(model_dir.rglob('*optim_states.pt'))):
            raise ValueError('Checkpoint optimizer save is incomplete')
    if (model_dir / 'model.safetensors.index.json').exists():
        kind = 'full'
        index = json.loads((model_dir / 'model.safetensors.index.json').read_text())
        names = sorted(set(index['weight_map'].values()))
        if not names or not all(name.endswith('.safetensors') for name in names):
            raise ValueError('invalid full-model weight index')
        if not (model_dir / 'tokenizer.json').is_file():
            raise ValueError('missing tokenizer in full-model export')
    elif (model_dir / 'model.safetensors').exists():
        kind = 'full'
        names = ['model.safetensors']
    elif (model_dir / 'adapter_model.safetensors').exists():
        kind = 'lora'
        names = ['adapter_model.safetensors']
        if not (model_dir / 'adapter_config.json').is_file():
            raise ValueError('missing adapter configuration')
    else:
        raise ValueError('not a full-model or LoRA final export')
    if kind == 'full' and not (model_dir / 'tokenizer.json').is_file():
        raise ValueError('missing tokenizer in full-model export')
    if base_model is not None:
        base_model = base_model.resolve(strict=True)
        if kind == 'lora':
            adapter = json.loads((model_dir / 'adapter_config.json').read_text())
            actual_base = Path(adapter['base_model_name_or_path']).resolve(strict=True)
            if actual_base != base_model:
                raise ValueError(f'adapter Base mismatch: {actual_base} != {base_model}')
        else:
            expected = json.loads((base_model / 'config.json').read_text())
            actual = json.loads((model_dir / 'config.json').read_text())
            keys = ('model_type', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                    'num_attention_heads', 'num_key_value_heads', 'vocab_size', 'tie_word_embeddings')
            mismatches = [key for key in keys if actual.get(key) != expected.get(key)]
            if mismatches:
                raise ValueError(f'full-model Base architecture mismatch: {mismatches}')
    files = {name: {'bytes': (model_dir / name).stat().st_size,
                    'sha256': digest(model_dir / name)} for name in names}
    return {'status': 'validated', 'kind': kind, 'run_id': run_id,
            'expected_base_model': str(base_model) if base_model is not None else None,
            'model_dir': str(model_dir), 'final_step': expected_step,
            'training_log_dir': str(log_dir), 'weights': files,
            'artifact_step': artifact_step, 'checkpoint_step': checkpoint_step,
            'training_exit_code': exit_code, 'parent_training_finished_successfully': exit_code == '0',
            'artifact_scope': 'complete_checkpoint' if checkpoint_step is not None else 'final_export'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, default=12208)
    parser.add_argument('--log-root', type=Path, default=LOG_ROOT)
    parser.add_argument('--base-model', type=Path,
                        help='Require a matching full-model architecture or exact LoRA Base identity')
    parser.add_argument('--checkpoint-step', type=int)
    parser.add_argument('--allow-running-checkpoint', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.model_dir, args.expected_step, args.log_root, args.base_model,
                           args.checkpoint_step, args.allow_running_checkpoint),
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
