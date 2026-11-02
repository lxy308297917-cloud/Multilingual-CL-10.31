"""Small reproducible benchmark records and resumable output; no inference framework."""
import hashlib,json,os,time
from pathlib import Path

SCHEMA='bench-v2.1'
STATUSES={'VERIFIED_COMPLETE','PILOT_ONLY','PARTIAL','NOT_RUN','FAILED'}

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def object_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def dump(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)

def jsonl(path):
 with Path(path).open() as stream:return [json.loads(line) for line in stream if line.strip()]

def checked_rows(path,expected=None):
 rows=jsonl(path);ids=[str(r['id']) for r in rows]
 if not rows or len(set(ids))!=len(ids):raise ValueError('Empty dataset or duplicate IDs')
 if expected is not None and len(rows)!=expected:raise ValueError('Unexpected sample count')
 return rows

def ids_hash(rows):return object_hash([str(r['id']) for r in rows])

def select_ids(rows,seed,count):
 return [r['id'] for r in sorted(rows,key=lambda r:hashlib.sha256((str(seed)+'|'+str(r['id'])).encode()).hexdigest())[:min(count,len(rows))]]

class PredictionStore:
 def __init__(self,output,manifest,expected_ids,resume=False):
  self.output=Path(output);self.expected=list(expected_ids)
  if len(set(self.expected))!=len(self.expected):raise ValueError('Duplicate expected IDs')
  identity=object_hash(manifest);self.rows={}
  if self.output.exists():
   if not resume:raise FileExistsError(self.output)
   old=json.loads((self.output/'manifest.json').read_text())
   if object_hash(old)!=identity:raise ValueError('Resume manifest differs: model/data/code/protocol changed')
   path=self.output/'predictions.jsonl'
   if path.exists():
    for r in jsonl(path):self._validate(r)
  else:
   self.output.mkdir(parents=True);dump(self.output/'manifest.json',manifest)
  self.path=self.output/'predictions.jsonl'
 def _validate(self,row):
  id=row['id']
  if id not in self.expected:raise ValueError('Unexpected prediction ID')
  if id in self.rows:raise ValueError('Duplicate prediction ID, even if identical')
  self.rows[id]=row
 def pending(self):return [id for id in self.expected if id not in self.rows]
 def append(self,rows):
  rows=list(rows);seen=set(self.rows)
  for r in rows:
   if r['id'] not in self.expected or r['id'] in seen:raise ValueError('Duplicate/unexpected output ID')
   seen.add(r['id'])
  with self.path.open('a') as f:
   for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n');self.rows[row['id']]=row
   f.flush();os.fsync(f.fileno())
 def finish(self,scope):
  missing=self.pending();status='PARTIAL' if missing else ('VERIFIED_COMPLETE' if scope=='full' else 'PILOT_ONLY')
  result=dict(result_schema_version=SCHEMA,run_status=status,samples=len(self.rows),expected=len(self.expected),missing_ids=missing,predictions_sha256=sha256(self.path) if self.path.exists() else None)
  dump(self.output/'integrity.json',result);return result

def require_complete(rows,expected_ids):
 ids=[r['id'] for r in rows]
 if len(set(ids))!=len(ids) or set(ids)!=set(expected_ids):raise ValueError('Incomplete or duplicate result cannot be scored as complete')
 by={r['id']:r for r in rows};return [by[id] for id in expected_ids]
