"""Few-probe real-model engineering gate; no efficacy or separation claims."""
import argparse,json,hashlib,os,subprocess,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from weights import load_pair,validate_pair,select_parameters,intervene,discover_components
from probes import load_probes
from inference import encode_probe,continuation_loglikelihood,evaluate_rows,summarize
from activations import capture


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--final',required=True);p.add_argument('--kind',choices=['full','lora'],required=True);p.add_argument('--probes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--gpu-uuid');a=p.parse_args()
    rows=load_probes(a.probes)
    if len(rows)>16 or any(r['source']['role']!='synthetic_smoke' for r in rows):raise ValueError('Engineering gate accepts at most16 explicitly synthetic probes')
    a.output.mkdir(parents=True,exist_ok=False)
    if a.device=='cuda':
        if not a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid:raise ValueError('Explicit single GPU UUID required')
        check=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'preflight_gpu.py'),'--uuid',a.gpu_uuid,'--max-user-gpus','4'],capture_output=True,text=True)
        (a.output/'gpu_preflight.json').write_text(check.stdout);check.check_returncode()
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42);tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    base,final=load_pair(a.base,a.final,a.kind,device=a.device);validate_pair(base,final)
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
    structure=discover_components(final);layer=len(structure)//2;selected=select_parameters(final,[layer],'attention')
    prefix,identity=encode_probe(tokenizer,rows[0]);answer=tokenizer.encode('A',add_special_tokens=False)
    baseline=continuation_loglikelihood(final,prefix,answer)
    with intervene(base,final,selected,'restore',1) as unchanged:
        during=continuation_loglikelihood(final,prefix,answer)
        assert baseline==during,'identity changes likelihood'
    assert unchanged['restoration_verified']
    with intervene(base,final,selected,'restore',0) as restored:
        lookup=dict(final.named_parameters());source=dict(base.named_parameters())
        assert all(torch.equal(lookup[n].detach().cpu(),source[n]) for n in selected)
        restored_score=continuation_loglikelihood(final,prefix,answer)
    assert restored['restoration_verified']
    after=continuation_loglikelihood(final,prefix,answer);assert after==baseline,'rollback changes likelihood'
    try:
        with intervene(base,final,selected,'restore',.5) as exceptional:raise RuntimeError('intentional smoke rollback')
    except RuntimeError as e:
        if str(e)!='intentional smoke rollback':raise
    assert exceptional['restoration_verified']
    repeat_rows=[r for r in rows if r['scoring']['kind']!='choice'][:2];assert len(repeat_rows)==2
    repeat=evaluate_rows(final,tokenizer,repeat_rows,max_new_tokens=16);repeat2=evaluate_rows(final,tokenizer,repeat_rows,max_new_tokens=16);assert repeat==repeat2
    module=structure[layer]['mlp'];ids=torch.tensor([prefix],device=a.device)
    with torch.inference_mode():
        with capture(final,[module]) as activations:final(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
    assert activations[module].shape[:2]==ids.shape and not dict(final.named_modules())[module]._forward_hooks
    results=evaluate_rows(final,tokenizer,rows,max_new_tokens=32)
    (a.output/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    record={'status':'engineering_gate_passed','scope':'Synthetic smoke only; no functional separation, task gain or benchmark claim','layer':layer,'component':'attention','identity_likelihood_exact':True,'rollback_likelihood_exact':True,'restore_parameters_equal_base':True,'exception_rollback_verified':True,'repeat_outputs_exact':True,'fixed_input_activation_capture_verified':True,'interventions':{'identity':unchanged,'restore':restored,'exception':exceptional},'restored_probe_loglikelihood':restored_score,'baseline_probe_loglikelihood':baseline,'probe_identity':identity,'samples':len(results),'probes_sha256':hashlib.sha256(a.probes.read_bytes()).hexdigest(),'chat_template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),'analysis_dtype':'float32','attention':'eager','enable_thinking':False,'do_sample':False,'use_model_defaults':False,'eos_token_id':tokenizer.eos_token_id,'elapsed_seconds':time.monotonic()-start,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else 0,'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if a.device=='cuda' else 0,'groups':summarize(results)}
    (a.output/'smoke.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:record[k] for k in ['status','samples','elapsed_seconds','gpu_peak_allocated_bytes']}))
if __name__=='__main__':main()
