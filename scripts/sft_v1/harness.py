"""Run upstream Harness unchanged; observe vLLM completion metadata only.

The upstream adapter returns text and drops finish reasons/token counts. This
observer keeps those diagnostics without changing prompts, outputs or scoring.
Single-process, single-GPU inference only (the frozen protocol uses DP=TP=1).
"""
import functools
import hashlib
import json
import os
from pathlib import Path


def install_trace(path):
    from vllm_compat import install
    install()
    from vllm import LLM
    path=Path(path)
    assert not path.exists(), 'Never append to another inference run'
    path.parent.mkdir(parents=True,exist_ok=True)
    original=LLM.generate
    def sha(value):
        return hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    @functools.wraps(original)
    def observed(self,*args,**kwargs):
        outputs=original(self,*args,**kwargs)
        records=[]
        for request in outputs:
            for completion in request.outputs:
                records.append({'request_id':request.request_id,'prompt_token_sha256':sha(list(request.prompt_token_ids)),'prompt_tokens':len(request.prompt_token_ids),'completion_index':completion.index,'response_sha256':sha(completion.text),'generated_tokens':len(completion.token_ids),'finish_reason':completion.finish_reason,'stop_reason':completion.stop_reason})
        with path.open('a') as stream:
            for row in records:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        return outputs
    LLM.generate=observed


if __name__=='__main__':
    install_trace(os.environ['SFT_VLLM_TRACE'])
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()
