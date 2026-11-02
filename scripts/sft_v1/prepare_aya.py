"""Bounded, source-stratified Aya candidates using the official LF tokenizer path.

No model outputs or formal scores enter selection. Run once into a new directory.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import unicodedata

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from llamafactory.hparams import DataArguments
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

ROOT = Path('/path/to/storage/datasets/qwen3-sft-v1-20261101')
OUT = ROOT / 'aya-sw-v2'
AUDIT = Path('/path/to/storage/datasets/qwen3-sft-source-audit-20261031')
MODEL = Path('/path/to/workspace/models/Qwen3-1.7B')
# Predeclared source policy: balance diverse instruction families; exclude
# templated event linking, simplification, completion and uncertain CoT mix.
SOURCES = ['Dolly-v2 (T)', 'SODA-inst (T)', 'Adversarial QA (T)',
           'Mintaka-inst (T)', 'HotpotQA (T)', 'CNN-Daily-Mail (T)',
           'Flan-GEM-wiki-lingua (T)', 'PIQA (T)', 'MasakhaNEWS-inst',
           'AfriSenti-inst', 'AfriQA-inst', 'Aya-Dataset', 'PAWS-Wiki (T)']


def norm(t):
    return ' '.join(unicodedata.normalize('NFKC', t).casefold().split())


def digest(t):
    return hashlib.sha256(t.encode()).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''): h.update(b)
    return h.hexdigest()


def main():
    assert not OUT.exists(), 'Preserve previous attempts; use a fresh version.'
    OUT.mkdir(parents=True)
    source_lock = json.loads(next((AUDIT/'CohereLabs--aya_collection_language_split').glob('*/SOURCE_LOCK.json')).read_text())
    path = Path(source_lock['path'])
    assert sha(path) == source_lock['sha256']
    rng = random.Random(20261101)
    reservoirs = {s: [] for s in SOURCES}
    counts = collections.Counter()
    # Reuse prior conservative anchor exclusions, without rescanning 3.58M texts.
    known = {int(l.split(':',1)[0])-1 for l in (AUDIT/'aya_collection_full_overlap_v1/anchor_hits.txt').read_text().splitlines()}
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            i = index; index += 1
            s = row['dataset_name']
            if s not in reservoirs or i in known: continue
            counts[s] += 1
            row['source_index'] = i
            pool = reservoirs[s]
            if len(pool) < 1400: pool.append(row)
            else:
                j = rng.randrange(counts[s])
                if j < len(pool): pool[j] = row
    candidates = [r for s in SOURCES for r in reservoirs[s]]
    # Existing account environment supplies GlotLID; do not mutate live train env.
    raw = OUT/'candidate_language_input.jsonl'
    raw.write_text(''.join(json.dumps({'inputs':r['inputs'],'targets':r['targets']},ensure_ascii=False)+'\n' for r in candidates))
    program = '''import fasttext,json,sys
m=fasttext.load_model('/path/to/workspace/models/glotlid/model_v3.bin')
for line in open(sys.argv[1]):
 r=json.loads(line);out={}
 for key in ['inputs','targets']:
  labs,ps=m.predict(r[key].replace('\\n',' ')[:6000],k=3)
  out[key]=list(zip(labs,map(float,ps)))
 print(json.dumps(out))
'''
    cp = subprocess.run(['/path/to/workspace/ssu-migration/envs/cl/bin/python','-c',program,str(raw)],capture_output=True,text=True,check=True)
    lids = [json.loads(l) for l in cp.stdout.splitlines()]
    assert len(lids) == len(candidates)
    (OUT/'candidate_language_results.jsonl').write_text(cp.stdout)
    # New benchmark inputs used only for exclusion, never for training/selection.
    anchors = []
    bench = ROOT/'benchmarks'
    locks = json.loads((bench/'sources.lock.json').read_text())
    for lock in locks.values():
        for rel in lock['files']:
            p = Path(lock['root'])/rel
            rows = pq.read_table(p).to_pylist() if p.suffix=='.parquet' else [json.loads(l) for l in p.read_text().splitlines()] if p.suffix=='.jsonl' else []
            for row in rows:
                q = row.get('question',row.get('prompt',''))
                if isinstance(q,str) and len(norm(q)) >= 40: anchors.append(norm(q))
    # Unique 80-character question anchors are a conservative exact exclusion.
    anchor80 = {q[:80] for q in anchors if len(q)>=80}
    (OUT/'benchmark_anchor_policy.json').write_text(json.dumps({'count':len(anchor80),'policy':'normalized first 80 chars anywhere in input/output; sources.lock.json identifies official files','limitation':'not semantic or exhaustive near-duplicate detection'},indent=2))
    blocked = set(); rejected = collections.Counter(); pools = collections.defaultdict(list)
    for row,lid in zip(candidates,lids):
        x,y=row['inputs'],row['targets'];nx,ny=norm(x),norm(y)
        if not nx or not ny or '\ufffd' in x+y or '<unk>' in x+y or len(x)>24000 or len(y)>14000:
            rejected['empty_corrupt_or_extreme']+=1;continue
        ph=digest(nx)
        if ph in blocked: rejected['duplicate_instruction']+=1;continue
        # Short labels and entity answers are exempt from unreliable response LID.
        if len(x.split())>=15 and not any('swh' in lab for lab,p in lid['inputs'][:1]):
            rejected['instruction_lid']+=1;continue
        if len(y.split())>=5 and not any('swh' in lab for lab,p in lid['targets'][:1]):
            rejected['response_lid']+=1;continue
        text=nx+' '+ny
        if any(text[i:i+80] in anchor80 for i in range(max(0,len(text)-79))):
            rejected['benchmark_anchor']+=1;continue
        blocked.add(ph)
        row['prompt_sha256']=ph; row['pair_sha256']=digest(nx+'\x1f'+ny)
        row['lid']=lid
        pools[row['dataset_name']].append(row)
    for pool in pools.values(): rng.shuffle(pool)
    # Round-robin source balance is identical for all prefix budgets.
    ordered=[]
    while any(pools.values()):
        for s in SOURCES:
            if pools[s]: ordered.append(pools[s].pop())
    assert len(ordered)>=6656, (len(ordered),dict(rejected))
    # Split by normalized instruction identity before tokenization or training.
    dev_sw=ordered[:256]; train=ordered[256:6656]
    human_path=next((AUDIT/'CohereLabs--aya_dataset').glob('*/*.parquet'))
    en=[dict(r,source_index=i) for i,r in enumerate(pq.read_table(human_path).to_pylist()) if r['language_code']=='eng' and len(r['targets'].split())>=15 and digest(norm(r['inputs'])) not in blocked]
    rng.shuffle(en); dev_en=en[:256]; assert len(dev_en)==256
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    args=DataArguments(template='qwen3',enable_thinking=False,train_on_prompt=False,cutoff_len=2048,packing=False)
    template=get_template_and_fix_tokenizer(tok,args)
    processor=SupervisedDatasetProcessor(template,tok,None,args)
    stats={};info={}
    for name,rows in [('pilot2000',train[:2000]),('train3200',train[:3200]),('train6400',train),('dev_sw',dev_sw),('dev_en',dev_en)]:
        features=[]; trunc=0
        for row in rows:
            msg=[{'role':'user','content':row['inputs']},{'role':'assistant','content':row['targets']}]
            ids,labels=processor._encode_data_example(msg[:1],msg[1:],None,None,[],[],[])
            src,tgt=template.encode_multiturn(tok,msg,None,None)[0]
            trunc+=len(src)+len(tgt)>2048
            assert len(ids)==len(labels)<=2048 and any(v==-100 for v in labels) and sum(v!=-100 for v in labels[1:])>0
            features.append({'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels})
        (OUT/f'{name}.jsonl').write_text(''.join(json.dumps({'instruction':r['inputs'],'output':r['targets']},ensure_ascii=False)+'\n' for r in rows))
        (OUT/f'{name}.provenance.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k not in ['inputs','targets']},ensure_ascii=False)+'\n' for r in rows))
        DatasetDict({'train':Dataset.from_list(features)}).save_to_disk(str(OUT/f'{name}_tokenized'))
        stats[name]={'rows':len(rows),'input_tokens':sum(len(f['input_ids']) for f in features),'shifted_assistant_tokens':sum(sum(v!=-100 for v in f['labels'][1:]) for f in features),'truncated':trunc,'sources':dict(collections.Counter(r.get('dataset_name','Aya-Dataset English') for r in rows)),'length_gt1024':sum(len(f['input_ids'])>1024 for f in features)}
        info[name]={'file_name':f'{name}.jsonl','columns':{'prompt':'instruction','response':'output'}}
    (OUT/'dataset_info.json').write_text(json.dumps(info,indent=2)+'\n')
    review=[]
    for s in SOURCES:review += [r for r in train if r['dataset_name']==s][:3]
    (OUT/'manual_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review))
    manifest={'source_lock':source_lock,'english_source':{'path':str(human_path),'sha256':sha(human_path)},'seed':20261101,'selection':'reservoir 1400/source; bounded quality+language filter; prompt dedup; source round robin; dev256 then nested train prefixes','excluded_source_families':'event-linking, simplification, lambada completion, CoT mix and miscellaneous sources not in explicit SOURCES','counts_seen':dict(counts),'candidate_count':len(candidates),'eligible':len(ordered),'rejected':dict(rejected),'stats':stats,'model':str(MODEL),'tokenizer_sha256':sha(MODEL/'tokenizer.json'),'cutoff_len':2048,'enable_thinking':False,'train_on_prompt':False,'dev_limitations':['Response loss is auxiliary, not benchmark ability. No reliable independent ability dev established yet. Use bounded manual generation review, disclose this limitation; never select from formal tests.','Source labels and GlotLID are fallible; short labels exempt from LID.','Normalized instruction grouping plus exact anchors do not guarantee semantic contamination absence.'],'files':{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file()}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'out':str(OUT),'eligible':len(ordered),'rejected':dict(rejected),'stats':stats},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
