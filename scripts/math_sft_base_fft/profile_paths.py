"""Paths for this scientific snapshot; configure local runtimes before execution."""
import os,sys
from pathlib import Path
ROOT=Path(os.environ.get('MLCL_ROOT',str(Path(__file__).resolve().parents[2])))
def resolve(key):
 defaults={
 'ROOT':ROOT,'DATA':ROOT/'data/math_sft_base_fft','MODEL':'/path/to/Qwen3-1.7B',
 'TRAIN_PY':sys.executable,'EVAL_PY':'/path/to/vllm-env/bin/python','CODE_PY':'/path/to/evalplus-env/bin/python',
 'POLY_ENV':'/path/to/polymath-scorer-env','AUDIT_PY':'/path/to/language-audit-env/bin/python',
 'CHECKPOINTS':ROOT/'checkpoints/math_sft_base_fft','LOGS':ROOT/'logs/math_sft_base_fft',
 'MMLU_TASKS':ROOT/'data/original_mmlu_tasks','CODE_DATA':ROOT/'data/bench_v2',
 'LEGACY_DATA':ROOT/'data/optional_legacy_unified','BENCH_CACHE':ROOT/'data/original_benchmark_cache',
 'SANDBOX_WORKSPACE':'/path/to/verified-sandbox-workspace','EXECUTOR':ROOT/'scripts/math_sft_base_fft/executor_adapter',
 'GLOTLID':'/path/to/glotlid/model_v3.bin','NLTK':'/path/to/frozen-nltk-resources','ENV_ROOT':'/path/to/envs','DATASET_ROOT':ROOT/'data','USER_STORAGE_ROOT':'/path/to/storage/user','SECONDARY_STORAGE_ROOT':'/path/to/storage/secondary'}
 env={'ROOT':'MLCL_ROOT','DATA':'MATH_SFT_DATA','MODEL':'MLCL_MODEL','TRAIN_PY':'MLCL_TRAIN_PY','EVAL_PY':'MLCL_EVAL_PY','CODE_PY':'MLCL_CODE_PY','POLY_ENV':'MLCL_POLYMATH_ENV','AUDIT_PY':'MLCL_AUDIT_PY','CHECKPOINTS':'MLCL_CHECKPOINT_DIR','LOGS':'MLCL_LOG_DIR','MMLU_TASKS':'MLCL_MMLU_TASKS','CODE_DATA':'MLCL_CODE_DATA_ROOT','LEGACY_DATA':'MLCL_LEGACY_DATA','BENCH_CACHE':'MLCL_BENCHMARK_CACHE','SANDBOX_WORKSPACE':'MLCL_SANDBOX_WORKSPACE','EXECUTOR':'MLCL_EXECUTOR_DIR','GLOTLID':'MLCL_GLOTLID','NLTK':'NLTK_DATA','ENV_ROOT':'MLCL_ENV_ROOT','DATASET_ROOT':'MLCL_DATASET_ROOT','USER_STORAGE_ROOT':'MLCL_USER_STORAGE_ROOT','SECONDARY_STORAGE_ROOT':'MLCL_SECONDARY_STORAGE_ROOT'}
 if key in ['TRAIN_ENV','CODE_ENV']:
  return str(Path(resolve('TRAIN_PY' if key=='TRAIN_ENV' else 'CODE_PY')).resolve().parent.parent)
 if key=='MATH_SOURCE':return os.environ.get('MLCL_MATHOCTOPUS_SOURCE',str(Path(resolve('DATA'))/'sources/mathoctopus/MGSM8KInstruct_Parallel.json'))
 if key=='GSM8K_TRAIN':return os.environ.get('MLCL_GSM8K_TRAIN',str(Path(resolve('DATA'))/'sources/gsm8k/train-00000-of-00001.parquet'))
 if key not in defaults:raise KeyError(key)
 return os.environ.get(env[key],str(defaults[key]))
