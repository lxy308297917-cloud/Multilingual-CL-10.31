"""Frozen single-example diagnostic inference; no benchmark score substitution."""
import hashlib,json,collections
import torch
from transformers import GenerationConfig
from probes import score_response


def encode_probe(tokenizer,row,max_input_tokens=1024):
    text=row['prompt']
    if row['scoring']['kind']=='choice':
        text+='\n'+'\n'.join(f'{label}. {answer}' for label,answer in zip('ABCD',row['choices']))
        text+='\n'+('Answer with one letter (A, B, C, D).' if row['input_language']=='en' else 'Jibu kwa herufi moja (A, B, C, D).')
    else:
        text+='\n'+({'en':{'en':'Respond in English.','sw':'Respond in Swahili.'},'sw':{'en':'Jibu kwa Kiingereza.','sw':'Jibu kwa Kiswahili.'}}[row['input_language']][row['output_language']])
    ids=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if not ids or len(ids)>max_input_tokens:raise ValueError('Probe exceeds frozen input budget; truncation is disallowed')
    return ids,{'rendered_user_prompt':text,'prompt_token_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'input_tokens':len(ids),'enable_thinking':False}


@torch.inference_mode()
def continuation_loglikelihood(model,prefix,continuation):
    if model.training:raise ValueError('Diagnostic inference requires eval mode')
    if not prefix or not continuation:raise ValueError('Nonempty prefix and continuation required')
    ids=torch.tensor([prefix+continuation],dtype=torch.long,device=next(model.parameters()).device)
    # Include the prefix's final position: first answer token is scored BEFORE
    # it is visible, while later answer tokens use their causal predecessors.
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(continuation)+1).logits
    aligned=logits[:,-len(continuation)-1:-1].float().log_softmax(-1)
    target=torch.tensor(continuation,device=ids.device).view(1,-1,1)
    values=aligned.gather(-1,target).squeeze(-1)
    return float(values.sum()),[float(v) for v in values[0]]


@torch.inference_mode()
def evaluate_rows(model,tokenizer,rows,max_new_tokens=128,max_input_tokens=1024,max_new_tokens_by_task=None):
    if model.training:raise ValueError('Diagnostic inference requires eval mode')
    device=next(model.parameters()).device;results=[]
    if tokenizer.eos_token_id is None:raise ValueError('Explicit chat EOS required')
    config=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=max_new_tokens,bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,use_cache=True)
    for row in rows:
        prefix,identity=encode_probe(tokenizer,row,max_input_tokens);scores=None;raw='';tokens=[]
        cap=(max_new_tokens_by_task or {}).get(row['task'],max_new_tokens)
        if not isinstance(cap,int) or cap<1:raise ValueError('Invalid frozen task generation cap')
        config.max_new_tokens=cap
        if row['scoring']['kind']=='choice':
            scores=[continuation_loglikelihood(model,prefix,tokenizer.encode(label,add_special_tokens=False))[0] for label in 'ABCD']
        else:
            inputs=torch.tensor([prefix],device=device);generated=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),generation_config=config,use_model_defaults=False)
            tokens=generated[0,len(prefix):].tolist();raw=tokenizer.decode(tokens,skip_special_tokens=True)
        results.append({'id':row['id'],'pair_id':row['pair_id'],'task':row['task'],'input_language':row['input_language'],'output_language':row['output_language'],'source':row['source'],'quality':row['quality'],**identity,'raw_response':raw,'generated_token_ids':tokens,'generated_tokens':len(tokens),'max_new_tokens':cap,'generation_hit_cap':len(tokens)==cap if scores is None else None,'score':score_response(row,raw,scores)})
    return results


def summarize(results):
    groups=collections.defaultdict(list)
    for r in results:groups[(r['task'],r['input_language'],r['output_language'])].append(r)
    summary=[]
    for key,rows in sorted(groups.items()):
        item=dict(task=key[0],input_language=key[1],output_language=key[2],samples=len(rows))
        for metric in ['content_correct','format_passed','language_correct']:
            values=[r['score'][metric] for r in rows if r['score'][metric] is not None]
            item[metric]={'correct':sum(values),'denominator':len(values),'rate':sum(values)/len(values) if values else None}
        item['option_distribution']=dict(collections.Counter(r['score']['predicted_option'] for r in rows if 'predicted_option' in r['score']))
        summary.append(item)
    return summary
