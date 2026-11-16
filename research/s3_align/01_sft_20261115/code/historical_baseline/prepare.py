"""Paired Bactrian EN/SW preparation; actual LLaMA-Factory tokenization."""
import collections,gzip,hashlib,json,os,random,unicodedata
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
from datasets import Dataset,DatasetDict
from transformers import AutoTokenizer
from llamafactory.hparams import DataArguments
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

HERE=Path(__file__).resolve().parent
OUT=HERE/'prepared-v1'
MODEL=Path('/srv/research/models/Qwen3-4B')
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def code(s):return '```' in s or any(t in s for t in ['def ','import ','function(','#include','SELECT ','console.log'])
def main():
    assert not OUT.exists(),'Use a new preparation version; never overwrite.'
    OUT.mkdir()
    policy=json.loads((HERE/'policy.json').read_text())
    raw={lang:json.load(gzip.open(HERE/'raw/data'/f'{lang}.json.gz','rt')) for lang in ['en','sw']}
    byid={}
    for lang,rows in raw.items():
        byid[lang]={r['id']:r for r in rows}
        assert len(byid[lang])==len(rows),'Duplicate raw source ID'
    lids={(r['language'],r['id']):r for r in map(json.loads,(HERE/'language_audit.jsonl').open())}
    patterns=Path('/srv/storage/datasets/qwen3-benchmark-exclusions-20261031/patterns.jsonl')
    anchors={norm(json.loads(line)['pattern'])[:80] for line in patterns.open() if len(norm(json.loads(line)['pattern']))>=80}
    # Include full FLORES devtest and exact target benchmark questions/passages.
    for lang in ['eng_Latn','swh_Latn']:
        p=Path('/srv/storage/datasets/flores200-meta-20261031/flores200_dataset/devtest')/f'{lang}.devtest'
        anchors.update(norm(s)[:80] for s in p.read_text().splitlines() if len(norm(s))>=80)
    reasons=collections.Counter();accepted=[];seen={lang:set() for lang in raw};audit=[]
    common=sorted(set(byid['en'])&set(byid['sw']))
    for ident in common:
        rejected=[];keys={}
        for lang in ['en','sw']:
            row=byid[lang][ident]
            x=row.get('instruction');y=row.get('output');inp=row.get('input') or ''
            if not isinstance(x,str) or not isinstance(y,str) or not isinstance(inp,str):
                rejected.append(lang+':invalid_fields');continue
            text=x+'\n\n'+inp+'\n\n'+y
            if not x.strip() or not y.strip() or '\ufffd' in text or '<unk>' in text or len(text)>100000:
                rejected.append(lang+':empty_corrupt_extreme');continue
            key=norm(x)+'\n'+norm(inp);keys[lang]=key
            if key in seen[lang]:rejected.append(lang+':duplicate_prompt')
            lid=lids[(lang,ident)]
            for field in ['instruction','output']:
                value=row[field];info=lid[field];expected='eng' if lang=='en' else 'swh'
                if len(value.split())>=20 and not code(value) and expected not in info['label'] and info['probability']>=0.9:
                    rejected.append(lang+':clear_wrong_language_'+field)
            nt=norm(text)
            if any(nt[i:i+80] in anchors for i in range(max(0,len(nt)-79))):rejected.append(lang+':exact_benchmark_anchor')
        if rejected:
            reasons.update(rejected);audit.append({'id':ident,'reasons':rejected});continue
        accepted.append(ident)
        for lang,key in keys.items():seen[lang].add(key)
    random.Random(policy['split_seed']).shuffle(accepted)
    dev=accepted[:500];train=accepted[500:]
    assert len(dev)==500 and len(train)>1000
    assert not set(dev)&set(train)
    (OUT/'ids.json').write_text(json.dumps({'train':train,'dev':dev},indent=2)+'\n')
    (OUT/'rejected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audit))
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    args=DataArguments(template='qwen3',enable_thinking=False,train_on_prompt=False,cutoff_len=2048,packing=False)
    template=get_template_and_fix_tokenizer(tok,args)
    processor=SupervisedDatasetProcessor(template,tok,None,args)
    info={};stats={};unusable=set();features={}
    for lang in ['en','sw']:
        for split,ids in [('train',train),('dev',dev)]:
            name=f'{lang}_{split}';feats=[];truncated=0;labels_total=0
            with (OUT/(name+'.jsonl')).open('x') as f:
                for ident in ids:
                    r=byid[lang][ident];prompt=r['instruction'].strip()
                    if r.get('input'):prompt+='\n\n'+r['input'].strip()
                    messages=[{'role':'user','content':prompt},{'role':'assistant','content':r['output'].strip()}]
                    tokens,labels=processor._encode_data_example(messages[:1],messages[1:],None,None,[],[],[])
                    src,tgt=template.encode_multiturn(tok,messages,None,None)[0]
                    truncated+=len(src)+len(tgt)>2048
                    assert len(tokens)==len(labels)<=2048
                    if not any(v==-100 for v in labels) or not any(v!=-100 for v in labels[1:]):unusable.add(ident)
                    feats.append({'input_ids':tokens,'attention_mask':[1]*len(tokens),'labels':labels})
                    f.write(json.dumps({'id':ident,'instruction':prompt,'output':r['output'].strip()},ensure_ascii=False)+'\n')
                    labels_total+=sum(v!=-100 for v in labels[1:])
            features[name]=feats
            stats[name]={'rows':len(feats),'input_tokens':sum(len(v['input_ids']) for v in feats),'shifted_assistant_tokens':labels_total,'truncated':truncated,'length_gt1024':sum(len(v['input_ids'])>1024 for v in feats)}
            info[name]={'file_name':name+'.jsonl','columns':{'prompt':'instruction','response':'output'}}
            print(name,stats[name],flush=True)
    assert not unusable,('Unsupervised truncated rows; paired filtering needs new preparation version',sorted(unusable))
    for name,feats in features.items():DatasetDict({'train':Dataset.from_list(feats)}).save_to_disk(str(OUT/(name+'_tokenized')))
    (OUT/'dataset_info.json').write_text(json.dumps(info,indent=2)+'\n')
    review=[{'id':ident,'en':byid['en'][ident],'sw':byid['sw'][ident]} for ident in dev[:12]]
    (OUT/'paired_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    manifest={'status':'PREPARED_PENDING_REVIEW','policy':policy,'policy_sha256':sha(HERE/'policy.json'),'source_revision':'3698480a001cb3f62c61c314d4acb181eb31e983','source_hashes':{lang:sha(HERE/'raw/data'/f'{lang}.json.gz') for lang in raw},'raw_rows':{k:len(v) for k,v in raw.items()},'common_ids':len(common),'accepted_pairs':len(accepted),'rejected_pairs':len(audit),'rejection_reasons':dict(reasons),'stats':stats,'model_path':str(MODEL),'tokenizer_sha256':sha(MODEL/'tokenizer.json'),'known_anchor_file_sha256':sha(patterns),'anchor_count':len(anchors),'limitations':['Outputs independently generated, not strict translated answer pairs.','GlotLID can err; short and code outputs exempt.','Exact 80-character anchor exclusion is not semantic contamination proof.','Same rows/epoch, unequal token exposure by language.'],'files':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'prepared':str(OUT),'accepted_pairs':len(accepted),'rejected_pairs':len(audit),'stats':stats},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
