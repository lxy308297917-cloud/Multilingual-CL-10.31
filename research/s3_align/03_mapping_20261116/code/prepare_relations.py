from common import *
import random,re,gzip
from transformers import AutoTokenizer
def encode(tok,text):
 ids=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 full=tok.decode(ids,skip_special_tokens=False);start=full.index(text);end=start+len(text)
 enc=tok(full,add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==ids
 positions=[i for i,(s,e) in enumerate(enc['offset_mapping']) if start<=s<e<=end]
 return dict(input_ids=ids,body_positions=positions)
def main():
 assert not (R/'inputs/relations64.jsonl').exists()
 paths={l:W/f'work/bactrian-single-20261112/raw/data/{l}.json.gz' for l in ['en','sw']}
 raw={l:{r['id']:r for r in json.load(gzip.open(p,'rt'))} for l,p in paths.items()}
 train_ids=set(read(DATA/'ids.json')['train']);align=rows(R/'inputs/alignment.jsonl')
 # Family exclusion is intentionally broader than the two diagnostic templates.
 pattern=r'\b(number|numbers|integer|integers|digit|digits|sort|sorting|sum|subtract|subtraction|ascending|descending|arithmetic|calculate|compute|addition|add|minus|plus)\b'
 excluded=[i for i,r in enumerate(align) if re.search(pattern,r['texts']['en'],re.I)]
 fit=[i for i in range(1024) if i not in excluded];assert len(fit)>=512
 tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True);rng=random.Random(20261116);result=[]
 for family in ['sorting','subtraction_order']:
  for j in range(32):
   if family=='sorting':
    numbers=rng.sample(range(100,999),4);values=', '.join(map(str,numbers))
    source_ids=['alpaca-54','alpaca-897'];assert all(s in train_ids for s in source_ids)
    orig_en=raw['en']['alpaca-54']['instruction']+'\n\n'+values
    orig_sw=raw['sw']['alpaca-54']['instruction']+'\n\n'+values
    pos_en='Sort these numbers from smallest to largest.\n\n'+values
    neg_en='Sort these numbers from largest to smallest.\n\n'+values
    neg_sw=raw['sw']['alpaca-897']['instruction']+'\n\n'+values
    # The SW paraphrase reuses a second genuine source sentence, changing only numbers.
    sw_para='Panga nambari zifuatazo kwa mpangilio wa kupanda: '+values+'.'
    source_ids.append('alpaca-845')
    answers={'positive':sorted(numbers),'negative':sorted(numbers,reverse=True)}
    transform='Numbers changed identically across languages; ascending/descending source templates reused. English positive/negative paraphrases matched except relation.'
   else:
    a,b=rng.sample(range(100,999),2);source_ids=['alpaca-36729'];assert source_ids[0] in train_ids
    en_template=raw['en'][source_ids[0]]['instruction'];sw_template=raw['sw'][source_ids[0]]['instruction']
    orig_en=en_template+f'\n\n{a}, {b}';orig_sw=sw_template+f'\n\n{a}, {b}'
    pos_en=f'What is the result of subtracting {b} from {a}?'
    neg_en=f'What is the result of subtracting {a} from {b}?'
    neg_sw=sw_template+f'\n\n{b}, {a}'
    sw_para=sw_template+f'\n\n{a}\n{b}'
    answers={'positive':a-b,'negative':b-a}
    transform='Same verified source bilingual instruction. Counterfactual swaps input operand order, preserves exact number multiset. SW synonym only separator change; English synonym is actual syntactic paraphrase.'
   assert answers['positive']!=answers['negative']
   texts=dict(en_original=orig_en,sw_original=orig_sw,en_positive=pos_en,en_negative=neg_en,sw_negative=neg_sw,sw_paraphrase=sw_para)
   result.append(dict(id=f'{family}-{j:02}',family=family,split='calibration_dev' if family=='sorting' else 'confirmation_provisional',source_ids=source_ids,texts=texts,answers=answers,quality='provisional_no_native_review',quality_details='Language templates copied from paired Bactrian-X sources; exact numeric transformations and answers verified programmatically. This is not human translation review.',transform=transform,encoded={k:encode(tok,v) for k,v in texts.items()}))
 jsonl(R/'inputs/relations64.jsonl',result)
 sources={sid:{l:raw[l][sid] for l in ['en','sw']} for r in result for sid in r['source_ids']}
 write(R/'inputs/relation_sources.json',sources)
 write(R/'records/RELATION_DATA_FROZEN.json',dict(at=now(),units=64,families=2,queries_per_unit=2,primary='SWoriginal: English same-meaning paraphrase vs relation-negative; SWnegative: reversed labels, balanced. Pair bootstrap clusters original unit, not128 independent rows.',fit_ids=[align[i]['id'] for i in fit],fit_indices=fit,excluded_family_indices=excluded,family_filter_regex=pattern,fit_scope='Task2 family-filtered subset of original training-side1024, neither diagnostic family used in fitting. No PCA/ridge/rank search.',training_mapping_scope='Training T1/T2 still uses frozen1024; its CCA applied separately to these relations as supplementary transfer check. Do not confuse with family-filtered diagnostic CCA.',split_by_family={'calibration_dev':'sorting32','confirmation_provisional':'subtraction_order32'},selection='No new outcome used for sample or method selection.',limitations=['All64 provisional without native human review; no strong natural-language semantic claim.','Only2 template families; numeric instances are not64 independent linguistic templates.','Source families may occur in historical main SFT; new numeric instances are held out from mapping fit, not certified pretraining-unseen.','Rule-based broad family exclusion, not exhaustive semantic family ontology.'],hashes={str(p):sha(p) for p in list(paths.values())+[R/'inputs/relations64.jsonl',R/'inputs/relation_sources.json']}))
 print('relations prepared',len(result),'family-filtered build',len(fit))
if __name__=='__main__':main()
