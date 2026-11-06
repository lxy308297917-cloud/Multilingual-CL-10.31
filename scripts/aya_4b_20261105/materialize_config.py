import argparse,os,json
from pathlib import Path
from profile_paths import resolve
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();assert not a.destination.exists();s=a.source.read_text()
for key in ['ROOT','APP','DATA','MODEL','LOGS','CHECKPOINTS','TRAIN_PY','EVAL_PY','CPU_ADAM_SO','BENCH_DATA','MMLU_TASKS','NLTK','LEGACY_DATA','USER_ROOT','SECONDARY_ROOT']:s=s.replace('${'+key+'}',resolve(key))
assert '${' not in s,'Unresolved variable';a.destination.parent.mkdir(parents=True,exist_ok=True);a.destination.write_text(s)
