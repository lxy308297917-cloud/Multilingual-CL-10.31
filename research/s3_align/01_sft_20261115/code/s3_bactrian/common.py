"""Independent Bactrian instruction track; historical math runs remain stopped."""
from pathlib import Path
import hashlib
import json

J = Path(__file__).resolve().parent
W = J.parents[1]
O = W/'outputs/s3_bactrian_20261113'
OLD = W/'work/bactrian-single-20261112'
MATH = W/'work/s3-align-v0-20261113'
BASE = Path('/srv/research/models/Qwen3-4B')
K = Path('/srv/storage/checkpoints/s3_bactrian_20261113')
CACHE = Path('/srv/storage/datasets/s3-bactrian-20261113/teacher-v1')
LOG = Path('/srv/research/logs/qwen3-multilingual/s3-bactrian-20261113')
PY = '/srv/storage/envs/qwen3-multilingual-20261031/bin/python'
REV = '3698480a001cb3f62c61c314d4acb181eb31e983'
LANGS = ['en','zh','fr','de','sw']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def dump(p,value):
    Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def read(p):return json.loads(Path(p).read_text())


def backbone(model):
    return (model.get_base_model() if hasattr(model,'get_base_model') else model).model


def pool(model,ids,positions):
    import torch
    saved=[]
    def hook(module,args,output):saved.append(output[0] if isinstance(output,tuple) else output)
    handle=backbone(model).layers[17].register_forward_hook(hook)
    try:
        backbone(model)(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    finally:handle.remove()
    assert len(saved)==1 and saved[0].shape[-1]==2560
    return saved[0][0,positions,:].float().mean(0)


def encode_prompt(tok,prompt):
    ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,
                                add_generation_prompt=True,enable_thinking=False)
    text=tok.decode(ids,skip_special_tokens=False)
    assert text.count(prompt)==1
    start=text.index(prompt);end=start+len(prompt)
    encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==ids
    positions=[i for i,(s,e) in enumerate(encoded['offset_mapping']) if start<=s<e<=end]
    assert positions
    return dict(input_ids=ids,body_positions=positions)
