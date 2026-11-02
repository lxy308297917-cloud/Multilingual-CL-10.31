"""Local-model independent probe evaluation CLI, one fixed protocol."""
import argparse,json,time,os,hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from weights import load_pair
from probes import load_probes
from inference import evaluate_rows,summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--final');p.add_argument('--kind',choices=['full','lora'],default='full');p.add_argument('--probes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--max-new-tokens',type=int,default=128);p.add_argument('--max-input-tokens',type=int,default=1024);a=p.parse_args()
    if a.device=='cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'):raise ValueError('Explicit preflighted GPU UUID required')
    rows=load_probes(a.probes);a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    if a.final:
        reference,model=load_pair(a.base,a.final,a.kind,device=a.device);del reference
    else:model=AutoModelForCausalLM.from_pretrained(a.base,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval().to(a.device)
    model.requires_grad_(False)
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
    results=evaluate_rows(model,tokenizer,rows,a.max_new_tokens,a.max_input_tokens)
    (a.output/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    info={'scope':'Independent diagnostic probes, not official benchmark scores','probes_sha256':hashlib.sha256(a.probes.read_bytes()).hexdigest(),'tokenizer_source':a.base,'chat_template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),'analysis_dtype':'float32','attention':'eager','do_sample':False,'use_model_defaults':False,'num_beams':1,'max_new_tokens':a.max_new_tokens,'max_input_tokens':a.max_input_tokens,'enable_thinking':False,'eos_token_id':tokenizer.eos_token_id,'samples':len(results),'elapsed_seconds':time.monotonic()-start,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else 0,'groups':summarize(results)}
    (a.output/'summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'samples':len(results),'elapsed_seconds':info['elapsed_seconds']}))
if __name__=='__main__':main()
