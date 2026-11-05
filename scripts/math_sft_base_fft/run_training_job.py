"""One two-GPU official training followed by CPU exposure/weight validation."""
import argparse, sys, subprocess, os, yaml
from pathlib import Path
from common import ROOT
ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
a = ap.parse_args()
script = Path(__file__).parent
target = Path(yaml.safe_load(Path(a.config).read_text())['output_dir'])
assert not target.exists() or not any(target.iterdir())
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2', '--standalone', '--nnodes=1', str(script / 'train_two_epochs.py'), '--config', a.config], cwd=ROOT, check=True)
subprocess.run([sys.executable, str(script / 'validate_training.py'), '--config', a.config], cwd=ROOT, check=True)
