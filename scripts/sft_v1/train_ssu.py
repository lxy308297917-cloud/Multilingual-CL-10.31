"""Thin SFT adaptation of the project's locked SSU-Wanda gradient hooks.

Uses official LLaMA-Factory SFT and the existing parameter-mask adapter.
ZeRO2 keeps parameter shapes available when the hooks are registered.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssu_mask_adapter import apply_gradient_masks
from train_formal import ExposureAccounting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stop-at', type=int)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    lock = json.loads((root / 'configs/mask_17b.lock.json').read_text())
    mask = Path(lock['mask_path'])
    assert hashlib.sha256(mask.read_bytes()).hexdigest() == lock['mask_sha256']
    config = yaml.safe_load(args.config.read_text())
    assert config['stage'] == 'sft' and config['finetuning_type'] == 'full'
    assert config['weight_decay'] == 0.0, 'Masked weights must not move through AdamW decay'
    ds = json.loads(Path(config['deepspeed']).read_text())
    assert ds['zero_optimization']['stage'] == 2
    if args.resume:
        config['resume_from_checkpoint'] = str(args.resume)
    import llamafactory.train.sft.workflow as workflow
    from llamafactory.train.tuner import run_exp

    original_load = workflow.load_model

    def load_masked(tokenizer, model_args, finetuning_args, is_trainable=False):
        model = original_load(tokenizer, model_args, finetuning_args, is_trainable)
        report = apply_gradient_masks(model, mask)
        assert report['total_unique_elements'] == lock['total_unique_elements']
        output = Path(config['output_dir'])
        output.mkdir(parents=True, exist_ok=True)
        (output / 'ssu-mask-applied.json').write_text(json.dumps({
            'mask_sha256': lock['mask_sha256'], 'mask_path': str(mask),
            'report': report, 'adaptation': 'SSU-Wanda hooks transferred from CPT to assistant-only SFT',
            'calibration_limit': 'Existing Tulu500 calibration is source-biased toward mathematics; not a new balanced calibration',
        }, indent=2) + '\n')
        print('SSU_MASK_REPORT', report, flush=True)
        return model

    workflow.load_model = load_masked
    run_exp(config, callbacks=[ExposureAccounting(args.resume, args.stop_at)])


if __name__ == '__main__':
    main()
