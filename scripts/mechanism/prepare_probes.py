"""Freeze independent diagnostic rows before any intervention/model scoring.

All sources are explicit local paths. Public train partitions are development
sources here, not claimed to be unseen by the pretrained language model.
"""
import argparse,ast,hashlib,json,re
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from probes import normalized,load_probes
from inference import encode_probe


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def order(value):return hashlib.sha256(str(value).encode()).hexdigest()
def read(path):
    with Path(path).open() as stream:return [json.loads(x) for x in stream if x.strip()]
def source(dataset,version,original_id,quality='source annotation; not independently reannotated'):
    return {'dataset':dataset,'version':version,'split':'train','original_id':str(original_id),'role':'independent_development'}, {'status':quality,'parallel_verified':False}
def row(id,pair,task,inp,out,prompt,answer,scoring,src,quality,choices=None):
    value=dict(id=id,pair_id=pair,task=task,input_language=inp,output_language=out,prompt=prompt,answer=answer,scoring=scoring,source=src,quality=quality)
    if choices is not None:value['choices']=choices
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--aya',type=Path,required=True);p.add_argument('--afriqa',type=Path,required=True);p.add_argument('--gsm-train',type=Path,required=True);p.add_argument('--baseline-results',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    train={normalized(x['instruction']) for x in read(a.aya/'train6400.jsonl')};dev={normalized(x['instruction']) for x in read(a.aya/'dev_sw.jsonl')};source_rows=read(a.afriqa);paired=[]
    for r in source_rows:
        if normalized(r['question']) not in dev or normalized(r['question']) in train:continue
        en=ast.literal_eval(r['translated_answer']);sw=ast.literal_eval(r['answers'])
        if not en or not sw or not all(isinstance(x,str) and x.strip() for x in en+sw):raise ValueError('Invalid heldout AfriQA answer')
        if r['translation_type']!='human_translation':raise ValueError('Require source human translation annotation')
        paired.append(dict(id=r['id'],en=r['translated_question'],sw=r['question'],answers={'en':en,'sw':sw}))
    assert len(paired)==22,'Unexpected AfriQA heldout identity'
    rows=[]
    for item in sorted(paired,key=lambda x:order(x['id'])):
        src,q=source('masakhane-io/afriqa','5c29933753692c28773c34ac1f68482007f1d2bb',item['id'],'Source labels human translation; spelling and factual annotation not independently revalidated');q['parallel_verified']=True;q['parallel_evidence']='Source bilingual questions and bilingual answer fields, shared source ID'
        options=[item]
        for other in sorted(paired,key=lambda x:order((item['id'],x['id']))):
            if other['id']==item['id']:continue
            if any(set(map(normalized,other['answers'][lang])).intersection(set(map(normalized,x['answers'][lang]))) for x in options for lang in ['en','sw']):continue
            options.append(other)
            if len(options)==4:break
        assert len(options)==4
        offset=int(order(item['id'])[:8],16)%4
        for rotation in [0,1]:
            shift=(offset+rotation)%4;arranged=options[shift:]+options[:shift];gold='ABCD'[[x['id'] for x in arranged].index(item['id'])]
            for lang in ['en','sw']:
                qual=dict(q,choice_negative_quality='Other heldout answers, alias-disjoint; semantic negative uniqueness is not independently verified',option_rotation=rotation)
                rows.append(row(f'knowledge-mc-{item["id"]}-{lang}-r{rotation}',f'afriqa-{item["id"]}','knowledge_choice',lang,lang,item[lang],gold,{'kind':'choice'},src,qual,[x['answers'][lang][0] for x in arranged]))
        if item in sorted(paired,key=lambda x:order(x['id']))[:8]:
            for inp in ['en','sw']:
                for out in ['en','sw']:
                    prompt=({'en':'Give only the short answer, without explanation.\n','sw':'Toa jibu fupi tu, bila maelezo.\n'}[inp])+item[inp]
                    rows.append(row(f'knowledge-qa-{item["id"]}-{inp}-{out}',f'afriqa-{item["id"]}','knowledge_short_answer',inp,out,prompt,item['answers'][out],{'kind':'aliases'},src,q))
    baseline=json.loads(a.baseline_results.read_text());metric_rows=baseline.get('rows',baseline.get('official_metric_rows'));demo_questions=set();demo_sources={}
    for r in metric_rows:
        if r['task']!='sft_v1_gsm8k':continue
        candidates=list(Path(r['official_result_file']).parent.glob('samples_sft_v1_gsm8k_*.jsonl'))
        if len(candidates)!=1:raise ValueError('Cannot uniquely find GSM demonstration trace')
        trace=candidates[0]
        if str(trace) in demo_sources:continue
        demo_sources[str(trace)]=sha(trace)
        for line in trace.open():
            doc=json.loads(line);text=doc['arguments']['gen_args_0']['arg_0'];questions=re.findall(r'<\|im_start\|>user\nQuestion: (.*?)\nAnswer:',text,re.S)
            if len(questions)!=6:raise ValueError('Expected five demonstrations plus actual question')
            demo_questions.update(normalized(x) for x in questions[:-1])
    assert demo_questions
    gsm=pq.read_table(a.gsm_train).to_pylist();eligible=[]
    for i,x in enumerate(gsm):
        if normalized(x['question']) in demo_questions or normalized(x['question']) in train:continue
        if len(tok.encode(x['answer'],add_special_tokens=False))>160:continue
        target=x['answer'].split('####')[-1].strip().replace(',','')
        if re.fullmatch(r'-?\d+(?:\.\d+)?',target) is None:continue
        eligible.append((i,x,target))
    for i,x,target in sorted(eligible,key=lambda x:order(x[0]))[:16]:
        src,q=source('openai/gsm8k','740312add88f781978c0658806c59bc2815b9866',i)
        rows.append(row(f'math-en-{i}',None,'math','en','en',x['question']+'\nEnd your answer with #### followed by the numerical result.',target,{'kind':'number'},src,q))
    sources=json.loads(a.source_manifest.read_text());kens=pq.read_table(sources['kenswquad_train']['path']).to_pylist();squad=json.loads(Path(sources['squad_train']['path']).read_text());english=[]
    for article in squad['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:english.append(dict(id=qa['id'],context=paragraph['context'],question=qa['question'],answers=[x['text'] for x in qa['answers']],story=article['title']))
    for lang,data,dataset,version in [('sw',kens,'Kencorpus/KenSwQuAD',sources['kenswquad_train']['revision']),('en',english,'Stanford/SQuAD','v1.1 official train')]:
        selected=0;stories=set()
        for item in sorted(data,key=lambda x:order(x['id'])):
            story=item.get('story_id',item.get('story'))
            if story in stories or normalized(item['question']) in train:continue
            answers=item.get('answers',[item.get('answer')])
            if not answers or any(not isinstance(x,str) or not x.strip() or x not in item['context'] for x in answers):continue
            prompt=({'en':'Read the passage and copy only the shortest answer span.\nPassage: ','sw':'Soma kifungu hiki na unakili jibu fupi tu.\nKifungu: '}[lang])+item['context']+'\n'+({'en':'Question: ','sw':'Swali: '}[lang])+item['question']
            src,q=source(dataset,version,item['id']);r=row(f'reading-{lang}-{item["id"]}',None,'reading_span',lang,lang,prompt,answers,{'kind':'aliases'},src,q)
            try:ids,_=encode_probe(tok,r,max_input_tokens=768)
            except ValueError:continue
            rows.append(r);stories.add(story);selected+=1
            if selected==16:break
        if selected!=16:raise ValueError('Insufficient independently sourced reading probes')
    # Original, transparently constructed instruction constraints; not IFEval.
    for i in range(8):
        for inp in ['en','sw']:
            for out in ['en','sw']:
                text=({'en':['all right','blue sky','good morning','red apple','thank you','small bird','green tree','clear water'],'sw':['sawa kabisa','anga bluu','habari asubuhi','tufaha jekundu','asante sana','ndege mdogo','mti kijani','maji safi']}[out])[i]
                prompt=({'en':'Write exactly these two words and nothing else: ','sw':'Andika maneno haya mawili tu bila kuongeza kitu kingine: '}[inp])+text
                src,q=source('project-original-constraints','20261102-v1',i,'Synthetic copy/word-count diagnostic; SW wording not independently reviewed');src['split']='constructed';q['parallel_verified']=False
                rows.append(row(f'instruction-{i}-{inp}-{out}',f'instruction-{i}','instruction_copy',inp,out,prompt,[text],{'kind':'constraints','constraints':[{'kind':'exact_words','value':2},{'kind':'fullmatch','value':re.escape(text)}],'language_reference_tokens':text.split()},src,q))
    for i in range(16):
        for lang in ['en','sw']:
            family=i%4
            if family==0:
                marker=['SUN','TRAIN','BOOK','CAT'][i//4];forbidden=['RAIN','CAR','PAGE','DOG'][i//4]
                prompt=({'en':f'Write exactly three UPPERCASE words separated by single spaces. Include the exact text {marker}. Do not include the exact text {forbidden}.','sw':f'Andika maneno matatu kwa HERUFI KUBWA, ukitenganisha kwa nafasi moja. Jumuisha maandishi {marker}. Usijumuishe maandishi {forbidden}.'}[lang])
                constraints=[{'kind':'exact_words','value':3},{'kind':'fullmatch','value':r'[A-Z]+(?: [A-Z]+){2}'},{'kind':'contains','value':marker},{'kind':'not_contains','value':forbidden}]
            elif family==1:
                key=['color','shape','animal','food'][i//4];value=['blue','circle','cat','rice'][i//4]
                prompt=({'en':f'Return only a JSON object with the single key "{key}" and value "{value}". No code block or extra text.','sw':f'Rudisha kitu cha JSON tu chenye ufunguo mmoja "{key}" na thamani "{value}". Usiweke maandishi mengine.'}[lang])
                constraints=[{'kind':'fullmatch','value':r'\{\s*"'+key+r'"\s*:\s*"'+value+r'"\s*\}'}]
            elif family==2:
                prompt=({'en':'Give exactly three different common nouns, one per line. Start each line with "- ". Include no heading or other text.','sw':'Toa nomino tatu tofauti za kawaida, moja kwa kila mstari. Anza kila mstari kwa "- ". Usiweke kichwa wala maandishi mengine.'}[lang])
                constraints=[{'kind':'fullmatch','value':r'- [^\s-]+\n- [^\s-]+\n- [^\s-]+'}]
            else:
                ending=['COMPLETE','READY','FINISH','DONE'][i//4]
                prompt=({'en':f'Write exactly two words. The last word must be {ending}. Do not use punctuation.','sw':f'Andika maneno mawili tu. Neno la mwisho liwe {ending}. Usitumie alama za uandishi.'}[lang])
                constraints=[{'kind':'exact_words','value':2},{'kind':'endswith','value':ending},{'kind':'fullmatch','value':r'[^\W\d_]+ [^\W\d_]+'}]
            src,q=source('project-original-multiconstraint','20261102-v1',i,'Author-constructed verifiable constraints; SW wording not independently reviewed; no factual content score');src['split']='constructed'
            rows.append(row(f'instruction-constraints-{i}-{lang}',f'instruction-constraints-{i}','instruction_constraints',lang,lang,prompt,None,{'kind':'constraints','constraints':constraints},src,q))
    for r in rows:encode_probe(tok,r,max_input_tokens=1024)
    file=a.output/'probes.jsonl';file.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));assert len(load_probes(file))==len(rows)
    import collections
    manifest={'selection':'Frozen before any model/intervention probe scoring; SHA256 source-ID ordering, not model scores','rows':len(rows),'task_counts':dict(collections.Counter(x['task'] for x in rows)),'file_sha256':sha(file),'afriqa_sha256':sha(a.afriqa),'aya_train_sha256':sha(a.aya/'train6400.jsonl'),'aya_dev_sw_sha256':sha(a.aya/'dev_sw.jsonl'),'gsm_train_sha256':sha(a.gsm_train),'gsm_demonstration_trace_hashes':demo_sources,'gsm_excluded_unique_demo_questions':len(demo_questions),'gsm_eligible_after_exclusion_and_reference_length':len(eligible),'reading_sources':sources,'quality_limits':['AfriQA source labels human translation, factual/spelling errors may remain','MC negatives alias-disjoint, semantic uniqueness not independently validated','Reading EN/SW are separate sources, NOT parallel','GSM reference <=160 tokens selected before scores; generation cap must be disclosed','Instruction copy probes are synthetic and narrow, not official IFEval','No claim sources unseen in original pretraining; no formal test questions copied to probes','Exact alias generation can fail due to surface form; preserve raw outputs and report format limits'],'official_test_for_selection':False}
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'rows':len(rows),'counts':manifest['task_counts'],'gsm_demonstration_exclusions':len(demo_questions)}))
if __name__=='__main__':main()
