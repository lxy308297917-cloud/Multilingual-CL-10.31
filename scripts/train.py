"""Use an existing LLaMA-Factory environment; never launch implicitly."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True, help='LLaMA-Factory YAML')
    parser.add_argument('--arm', choices=['T0', 'T1', 'T2'], required=True)
    parser.add_argument('--assets', type=Path, required=True, help='Frozen asset deployment JSON')
    parser.add_argument('--check-only', action='store_true', help='CPU config/asset validation; no training')
    args = parser.parse_args()
    import yaml
    from s3_align.assets import load_assets, sha256
    cfg = yaml.safe_load(args.config.read_text())
    out = Path(cfg['output_dir'])
    assert not out.exists(), 'Use a new output directory; no implicit resume or overwrite'
    assert cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps'] == 16
    assert cfg['seed'] == 42 and cfg['use_reentrant_gc'] is False
    assert not cfg.get('adapter_name_or_path') and not cfg.get('resume_from_checkpoint')
    assets = load_assets(args.assets, args.arm)
    if args.check_only:
        print(json.dumps({'arm': args.arm, 'asset_validation': 'passed', 'training_started': False}))
        return
    from s3_align.accounting import ExposureAccounting
    from s3_align.trainer import make_trainer
    import llamafactory.train.sft.workflow as workflow
    from llamafactory.train.tuner import run_exp
    previous = workflow.CustomSeq2SeqTrainer
    workflow.CustomSeq2SeqTrainer = make_trainer(previous, assets, args.arm)
    try:
        run_exp(cfg, callbacks=[ExposureAccounting()])
    finally:
        workflow.CustomSeq2SeqTrainer = previous
    (out / 'recipe_identity.json').write_text(json.dumps({
        'arm': args.arm, 'config_sha256': sha256(args.config),
        'assets_spec_sha256': sha256(args.assets),
        'implementation': 'organized-development-version; not the historical frozen runner',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
