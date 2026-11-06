"""Explicit paths for portable scientific reproduction."""
import os,sys
from pathlib import Path
ROOT=Path(os.environ.get('MLCL_ROOT',str(Path(__file__).resolve().parents[2])))
def resolve(key):
 defaults={'ROOT':ROOT,'APP':ROOT,'DATA':ROOT/'data/aya_4b_20261105','MODEL':ROOT/'models/Qwen3-4B','LOGS':ROOT/'logs/aya_4b_20261105','CHECKPOINTS':ROOT/'checkpoints/aya_4b_20261105','TRAIN_PY':sys.executable,'EVAL_PY':sys.executable,'CPU_ADAM_SO':'/path/to/optional/prebuilt/cpu_adam.so','BENCH_DATA':ROOT/'data/bench_v2','MMLU_TASKS':ROOT/'data/mmlu_tasks','NLTK':ROOT/'data/nltk','LEGACY_DATA':ROOT/'data/optional_legacy','USER_ROOT':ROOT/'local','SECONDARY_ROOT':ROOT/'local_secondary'}
 if key not in defaults:raise KeyError(key)
 return os.environ.get('AYA4B_'+key,str(defaults[key]))
