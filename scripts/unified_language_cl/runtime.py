"""Deployment paths only; model/data/scoring definitions live in frozen protocols."""
import os
import sys
from pathlib import Path
PROJECT=Path(os.environ.get("MLCL_PROJECT_ROOT",Path(__file__).resolve().parents[2]))
DEFAULTS={"retention_lock":PROJECT/"runs/reproduction/retention/stack.lock.json","retention_core_tasks":PROJECT/"runs/reproduction/retention/core_tasks","retention_reading_tasks":PROJECT/"runs/reproduction/retention/reading_tasks","belebele_source":PROJECT/"data/benchmarks/belebele","ifeval_source":PROJECT/"data/benchmarks/ifeval","gsm8k_source":PROJECT/"data/gsm8k/main","project":PROJECT,"data":PROJECT/"data/unified_language_cl","model":PROJECT/"models/Qwen3-1.7B","checkpoints":PROJECT/"runs/reproduction/checkpoints","logs":PROJECT/"runs/reproduction/logs","results":PROJECT/"runs/reproduction/results","configs":PROJECT/"runs/reproduction/configs","train_python":sys.executable,"eval_python":sys.executable,"old_eval_data":PROJECT/"data/old_eval_locked","old_nltk":PROJECT/"data/nltk_data","rouge_source":PROJECT/"vendor/xlsum-afcd803c8d7c98f3be60e9a7ce57fcbc3e729e8c","metrics_addons":PROJECT/".venv-metrics","glotlid":PROJECT/"models/glotlid"}
def runtime_path(key,relative=""):
    base=Path(os.environ.get("MLCL_"+key.upper(),DEFAULTS[key]))
    if key not in ("train_python","eval_python"):base=base.expanduser().resolve()
    result=str(base/relative.lstrip("/")) if relative else str(base)
    return result+("/" if relative.endswith("/") or relative=="/" else "")
