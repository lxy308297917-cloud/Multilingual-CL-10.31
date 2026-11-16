"""Forward the complete CLI to upstream Harness without defining a new scorer."""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
if __name__ == '__main__':
    # Optional historical trace hook, only when explicitly requested.
    if os.environ.get('SFT_VLLM_TRACE'):
        runpy.run_module('s3_align.harness', run_name='__main__')
    else:
        from lm_eval.__main__ import cli_evaluate
        cli_evaluate()
