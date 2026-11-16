from pathlib import Path
import json, hashlib, datetime
R=Path(__file__).resolve().parents[1]
HOME=Path('/srv/research')
W=HOME/'Documents/Codex/2026-11-10/new-chat'
P=HOME/'10.31/multilingual-research'
PREV=HOME/'reproductions/s3_mechanism_followup_20261115_v1'
OLD=HOME/'reproductions/shared_subspace_analysis_20261115'
ANCHOR=HOME/'reproductions/sw_function_anchor_20261115'
BASE=HOME/'models/Qwen3-4B'
CK=HOME/'models/checkpoint/s3_mapping_20261116'
LOG=HOME/'logs/s3_mapping_20261116'
TP='/srv/storage/envs/qwen3-multilingual-20261031/bin/python'
EP='/srv/storage/envs/qwen3-eval-vllm-20261101/bin/python'
DATA=W/'work/bactrian-single-20261112/prepared-v1'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def jsonl(p,x):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
