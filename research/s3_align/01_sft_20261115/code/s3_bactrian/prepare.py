"""Reuse full SFT stream; construct a small independent instruction alignment pool."""
import collections
import gzip
import json
import os
from common import *


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    O.mkdir(parents=True,exist_ok=True)
    assert not (J/'prepared').exists(),'Preserve prepared versions'
    manifest=read(OLD/'prepared-v1/manifest.json')
    for filename,digest in manifest['files'].items():
        assert sha(OLD/'prepared-v1'/filename)==digest,filename
    assert sha(K.parent/'bactrian_single_20261112/sw_final_v1/adapter_model.safetensors')=='3988161e1a60e35f1ec08be9f20e138e4a9c83b29be80c6defe0345ae5d5f003'
    raw={};source={}
    for lang in LANGS:
        path=OLD/'raw/data'/f'{lang}.json.gz'
        if not path.exists():
            path=Path(hf_hub_download('MBZUAI/Bactrian-X',f'data/{lang}.json.gz',
                repo_type='dataset',revision=REV,local_dir=J/'raw'))
        rows=json.load(gzip.open(path,'rt'))
        raw[lang]={r['id']:r for r in rows};assert len(raw[lang])==len(rows)
        source[lang]=dict(path=str(path),sha256=sha(path),rows=len(rows),revision=REV)
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    ids=read(OLD/'prepared-v1/ids.json')
    selected=[];rejections=collections.Counter()
    for ident in ids['train']:
        encoded={};texts={};bad=False
        for lang in LANGS:
            r=raw[lang].get(ident)
            if r is None or not isinstance(r.get('instruction'),str) or not r['instruction'].strip():
                rejections['missing_or_empty_reference']+=1;bad=True;break
            inp=r.get('input') or ''
            if not isinstance(inp,str):rejections['invalid_input']+=1;bad=True;break
            prompt=r['instruction'].strip()+ ('\n\n'+inp.strip() if inp else '')
            e=encode_prompt(tok,prompt)
            if len(e['input_ids'])>512:
                rejections['auxiliary_prompt_longer_than_512']+=1;bad=True;break
            encoded[lang]=e;texts[lang]=prompt
        if not bad:
            selected.append(dict(id=ident,encoded=encoded,texts=texts))
        if len(selected)==1024:break
    assert len(selected)==1024 and not {r['id'] for r in selected}&set(ids['dev'])
    p=J/'prepared';p.mkdir()
    (p/'alignment.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    dump(p/'manifest.json',dict(status='PREPARED_ALIGNMENT_POOL_FROM_TRAIN_ONLY',
        source=source,alignment_rows=1024,ordered_ids=[r['id'] for r in selected],
        alignment_selection='First 1024 in frozen historical train-ID order passing nonempty/reference-ID/prompt<=512 checks, all five languages. No benchmark use.',
        exclusions_before_1024=dict(rejections),main_sft_unchanged=True,
        main_sft_rows=65415,main_sft_supervised_tokens=14314363,
        historical_manifest_sha256=sha(OLD/'prepared-v1/manifest.json'),
        alignment_sha256=sha(p/'alignment.jsonl'),
        scope='Auxiliary instruction-only pool; it does not filter, shorten, or replace the full 65415-row SFT stream. Reference answers never enter alignment.',
        approximation='Bactrian same-ID instructions are translated counterparts; output texts are independently generated and not used as parallel answers.'))
    print(json.dumps(dict(status='PREPARED',alignment_rows=1024,sft_rows=65415,source=source)),flush=True)


if __name__=='__main__':main()
