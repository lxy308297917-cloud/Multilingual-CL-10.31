"""Same-input prompt activation interventions on teacher-forced answer likelihood.

Not a generated-answer benchmark. No cross-language token alignment. Donors
and recipients always use identical input IDs except explicit different-item
same-language/length controls, whose positions have no semantic alignment.
"""
import argparse,collections,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from weights import load_pair,discover_components
from probes import load_probes
from inference import encode_probe,continuation_loglikelihood
from activations import capture,patch


def reference_text(row):
    answer=row['answer']
    if answer is None:return None
    text=answer[0] if isinstance(answer,list) else str(answer)
    if row['scoring']['kind']=='number':
        from probes import score_response
        text=row['scoring'].get('reference_template','#### {answer}').format(answer=text)
        score=score_response(row,text)
        if not score['content_correct'] or not score['format_passed']:raise ValueError('Reference continuation violates numeric content/format rule')
    return text


def reference_tokens(tokenizer,row):
    text=reference_text(row)
    if text is None:return None
    ids=tokenizer.encode(text,add_special_tokens=False)
    if not ids:raise ValueError('Empty reference continuation')
    return ids


@torch.inference_mode()
def prompt_activations(model,prefix,answer,names):
    # Capture from the same complete teacher-forced input shape used by LL.
    ids=torch.tensor([prefix+answer],device=next(model.parameters()).device)
    with capture(model,names,list(range(len(prefix)))) as result:model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
    return result


def norm_matched_noise(recipient,donor,seed):
    if recipient.shape!=donor.shape:raise ValueError('Activation shapes differ')
    delta=donor-recipient;generator=torch.Generator(device='cpu');generator.manual_seed(seed);noise=torch.randn(recipient.shape,generator=generator,dtype=recipient.dtype)
    scale=torch.linalg.vector_norm(delta.double())/torch.linalg.vector_norm(noise.double());return recipient+noise*scale


def neutral_control_prefix(tokenizer,language,length):
    if language not in ('en','sw'):raise ValueError('Unsupported control language')
    anchor=tokenizer.encode('z',add_special_tokens=False)
    if len(anchor)!=1:raise ValueError('Need a unique single-token template anchor')
    scaffold=tokenizer.apply_chat_template([{'role':'user','content':'z'}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    positions=[i for i,x in enumerate(scaffold) if x==anchor[0]]
    if len(positions)!=1:raise ValueError('Cannot isolate user-content template span')
    position=positions[0];budget=length-(len(scaffold)-1)
    sentence={'en':'ZORBU is a stone.','sw':'ZORBU ni jiwe.'}[language]
    tokens=tokenizer.encode(sentence,add_special_tokens=False);dot=tokenizer.encode('.',add_special_tokens=False)
    if not tokens or len(dot)!=1 or budget<len(tokens):raise ValueError('Control prefix too short or cannot pad with a single period token')
    repeats,remainder=divmod(budget,len(tokens));content=tokens*repeats+dot*remainder
    prefix=scaffold[:position]+content+scaffold[position+1:]
    assert len(prefix)==length
    return prefix,{'language':language,'sentence':sentence,'sentence_repeats':repeats,'period_padding_tokens':remainder,'prompt_tokens':length,'token_level_length_matched':True,'control_token_sha256':hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest(),'semantics':'Synthetic fictional entity absent from source probes; no task answer information','quality':'Repeated neutral clause plus period padding can be OOD; not distribution matched. SW clause not independently reviewed.'}


def patched_ll(model,prefix,answer,name,donor):
    with patch(model,name,donor,list(range(len(prefix)))):return continuation_loglikelihood(model,prefix,answer)[0]


def choose_cases(tokenizer,rows,max_per_group):
    encoded={r['id']:encode_probe(tokenizer,r)[0] for r in rows};groups=collections.defaultdict(list)
    for r in rows:
        if r['answer'] is not None:groups[(r['task'],r['input_language'],r['output_language'])].append(r)
    cases=[]
    for key,group in sorted(groups.items()):
        for r in sorted(group,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:max_per_group]:
            candidates=[q for q in rows if q['input_language']==r['input_language'] and len(encoded[q['id']])==len(encoded[r['id']]) and (q['pair_id'] or q['id'])!=(r['pair_id'] or r['id'])]
            candidates.sort(key=lambda q:(q['task']==r['task'],hashlib.sha256(q['id'].encode()).hexdigest()))
            cases.append((r,candidates[0] if candidates else None))
    return cases,encoded


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--final',required=True);p.add_argument('--kind',choices=['full','lora'],required=True);p.add_argument('--probes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--layer',type=int,default=14);p.add_argument('--max-per-group',type=int,default=2);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--gpu-uuid');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.device=='cuda':
        if not a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid:raise ValueError('Explicit single UUID required')
        check=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'preflight_gpu.py'),'--uuid',a.gpu_uuid,'--max-user-gpus','4'],capture_output=True,text=True);(a.output/'gpu_preflight.json').write_text(check.stdout);check.check_returncode()
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42);tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True);rows=load_probes(a.probes);cases,encoded=choose_cases(tokenizer,rows,a.max_per_group);base,final=load_pair(a.base,a.final,a.kind,device=a.device);base.to(a.device);structure=discover_components(final)
    if a.layer<0 or a.layer>=len(structure):raise ValueError('Layer outside model')
    others=[i for i in range(len(structure)) if i!=a.layer];random_layer=random.Random(42).choice(others);names=[structure[i][c] for i in [a.layer,random_layer] for c in ['attention','mlp']];metadata={structure[i][c]:(i,c) for i in [a.layer,random_layer] for c in ['attention','mlp']};records=[]
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
    for row,unrelated in cases:
        prefix=encoded[row['id']];answer=reference_tokens(tokenizer,row);gold=continuation_loglikelihood(final,prefix,answer)[0];source_gold=continuation_loglikelihood(base,prefix,answer)[0];source=prompt_activations(base,prefix,answer,names);recipient=prompt_activations(final,prefix,answer,names)
        neutral_prefix,neutral_info=neutral_control_prefix(tokenizer,row['input_language'],len(prefix));neutral=prompt_activations(base,neutral_prefix,[answer[0]],names)
        if 'ZORBU' in row['prompt']:raise ValueError('Neutral entity occurs in target prompt')
        other=None
        if unrelated is not None:
            # The other item's own answer is not injected. A neutral single-token
            # continuation keeps this controlled capture causal and shape-bounded.
            other=prompt_activations(base,encoded[unrelated['id']],[answer[0]],names)
        for name in names:
            layer,component=metadata[name];identity=patched_ll(final,prefix,answer,name,recipient[name])
            if abs(identity-gold)>1e-5*max(1,len(answer)):raise ValueError('Identity activation patch changes answer likelihood')
            seed=int(hashlib.sha256((row['id']+name).encode()).hexdigest()[:8],16);noise=norm_matched_noise(recipient[name],source[name],seed)
            donors=[('same_input_base',source[name]),('random_activation_norm_matched',noise),('synthetic_neutral_semantic',neutral[name])]
            if other is not None:donors.append(('different_item_same_language_length',other[name]))
            for condition,donor in donors:
                actual=patched_ll(final,prefix,answer,name,donor);norm=float(torch.linalg.vector_norm((donor-recipient[name]).double()))
                records.append({'id':row['id'],'pair_id':row['pair_id'],'task':row['task'],'input_language':row['input_language'],'output_language':row['output_language'],'layer':layer,'component':component,'layer_role':'predeclared_middle' if layer==a.layer else 'seed42_random_control','condition':condition,'reference_token_count':len(answer),'prompt_tokens':len(prefix),'final_reference_loglikelihood':gold,'base_reference_loglikelihood':source_gold,'patched_reference_loglikelihood':actual,'patched_minus_final_loglikelihood':actual-gold,'per_token_loglikelihood_change':(actual-gold)/len(answer),'identity_absolute_error':abs(identity-gold),'perturbation_norm':norm,'same_input_base_perturbation_norm':float(torch.linalg.vector_norm((source[name]-recipient[name]).double())),'different_item_control_id':unrelated['id'] if unrelated is not None else None,'different_item_control_task':unrelated['task'] if unrelated is not None else None,'control_semantics_note':'Different item, prioritizes different task, not independently guaranteed unrelated meaning; same-language shape control, not semantic position alignment','gold_answer_not_used_in_prompt_patch':True,'neutral_control_metadata':neutral_info})
    (a.output/'patching_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));summary={'status':'completed','reference_format':'Numeric continuation uses requested #### result; first accepted alias for other scored answers','version':3,'scope':'Teacher-forced reference LL, no generated-answer benchmark or broad causal attribution. Only prompt positions patched. No cross-language activation patch.','probe_sha256':hashlib.sha256(a.probes.read_bytes()).hexdigest(),'selection':'First SHA256(id) per task/language group, no model-score selection','selected_ids':[r['id'] for r,_ in cases],'samples':len(cases),'neutral_semantic_control_available':len(cases),'different_item_control_available':sum(q is not None for _,q in cases),'missing_control_reason':'No same-language identical-prompt-length different item; no interpolation or cross-language token mapping','middle_layer':a.layer,'random_layer':random_layer,'records':len(records),'elapsed_seconds':time.monotonic()-start,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else 0,'functional_separation_established':False,'limitations':['First supplied answer alias only, not total probability over all valid answers','Answer LL can improve without downstream accuracy or instruction-following gain','Different-item controls may remain semantically related; report availability and OOD incompatibility','No gold reference for free-form instruction constraints; those probes excluded from this LL experiment','Noise matches total perturbation norm, not the activation distribution','Synthetic neutral-semantic controls use a repeated fictional noun clause and period padding to match token length; semantic mismatch is intentional, distribution shift remains']};(a.output/'DONE.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
