"""Freeze a project-fresh human-translated ARC train cohort before model output."""
from pathlib import Path
import argparse, collections, datetime, hashlib, json, re, sys, unicodedata
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from lm_eval.api.model import TemplateLM


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())


def main():
    p=argparse.ArgumentParser()
    for n in ['source','project','output','tokenizer','ha-train','sw-train','previous-mmlu','previous-dev','previous-reading']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    sys.path.insert(0,str(a.project/'scripts/answer_bias_study'))
    from prepare import invalid
    langs=['en','sw','ha'];maps={};sources={}
    for l in langs:
        f=a.source/(l+'_train.json');sources[l+'_arc_train']=sha(f)
        rows=json.loads(f.read_text());maps[l]={v['id']:v for v in rows};assert len(rows)==len(maps[l])
    common=set.intersection(*(set(v) for v in maps.values()))
    exposed=set()
    for f in [a.previous_mmlu,a.previous_dev]:
        sources['previous_'+f.parent.name+'_'+f.name]=sha(f)
        for v in pq.read_table(f).to_pylist():exposed.add(norm(v['question']))
    sources['previous_reading']=sha(a.previous_reading)
    for v in map(json.loads,a.previous_reading.open()):exposed.add(norm(v['question']))
    training=[]
    for f in [a.ha_train,a.sw_train]:
        sources['training_'+f.parent.name]=sha(f)
        for v in map(json.loads,f.open()):
            text='\n'.join(str(v.get(k,'')) for k in ['instruction','input','output']);training.append(norm(text))
    visual=re.compile(r'\b(diagram|graph|picture|figure|table|map|shown|illustration|picha|mchoro|jedwali|ramani|grafu|zane|taswira|hoton|jadawali)\b',re.I)
    excluded=[];eligible={};seen=set()
    for sid in sorted(common):
        triple=[];reasons=[]
        for l in langs:
            v=maps[l][sid];labels=v['choices']['label'];choices=v['choices']['text']
            if len(choices)!=4 or len(labels)!=4:reasons.append('not_four');continue
            gold=labels.index(v['answerKey']);reasons.extend([invalid(choices)] if invalid(choices) else [])
            if visual.search(v['question']):reasons.append('requires_missing_visual_or_visual_reference')
            if norm(v['question']) in exposed:reasons.append('exact_previous_benchmark_question')
            q=norm(v['question'])
            if len(q)>=24 and any(q in text for text in training):reasons.append('direct_current_SFT_question_substring')
            triple.append(dict(id=sid,cluster=sid,language=l,task='uhura_arc_train',subject='science',question=v['question'].strip(),choices=choices,gold=gold))
        if len(triple)!=3 or len(set(v['gold'] for v in triple))!=1:reasons.append('gold_or_language_alignment')
        identity=norm(maps['en'][sid]['question'])
        if identity in seen:reasons.append('duplicate_english_question')
        seen.add(identity)
        if reasons:excluded.append(dict(id=sid,reasons=sorted(set(reasons))));continue
        eligible[sid]=triple
    seed='decision-stability-fresh-arc-20261104-v1'
    ids=sorted(eligible,key=lambda k:hashlib.sha256((seed+'|'+k).encode()).hexdigest())[:400];assert len(ids)==400,(len(ids),len(eligible))
    split_ids={'train_aux':ids[:160],'dev':ids[160:280],'holdout':ids[280:]}
    assert not any(set(v)&set(w) for k,v in split_ids.items() for q,w in split_ids.items() if k!=q)
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    class Encoder:
        backend='causal'
        def tok_encode(self,text,**kwargs):return tok.encode(text,add_special_tokens=False)
    encoder=Encoder()
    def encode(text,answers):
        prompt=tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        pairs=[TemplateLM._encode_pair(encoder,prompt,s) for s in answers]
        assert all(c and t and len(c)+len(t)<=2048 for c,t in pairs)
        return pairs
    file_hashes={};counts={};maxlen=0
    for split,sids in split_ids.items():
        output=[]
        for sid in sids:
            for x in eligible[sid]:
                rotations=[];controls=[];base='Q: '+x['question']+'\n'
                for r in range(4):
                    order=[(i+r)%4 for i in range(4)];opts=[x['choices'][i] for i in order]
                    for shift in range(4):
                        label_order=[(i+shift)%4 for i in range(4)]
                        text=base+'\n'.join('ABCD'[lab]+': '+opt for lab,opt in zip(label_order,opts))+'\nAnswer:'
                        pairs=encode(text,list('ABCD'));maxlen=max(maxlen,max(len(c)+len(t) for c,t in pairs))
                        # Candidates are always token IDs A/B/C/D; recover content and physical location separately.
                        label_to_position=[label_order.index(j) for j in range(4)]
                        label_to_content=[order[pos] for pos in label_to_position]
                        if shift:
                            if split!='train_aux':controls.append(dict(rotation=r,label_shift=shift,label_to_position=label_to_position,label_to_content=label_to_content,letter_pairs=pairs))
                        else:
                            content=base+'Options:\n'+'\n'.join('- '+s for s in opts)+'\nAnswer with the complete text of the correct option.\nAnswer:'
                            cp=encode(content,opts)
                            rotations.append(dict(rotation=r,order=order,gold_position=order.index(x['gold']),letter_pairs=pairs,content_pairs=cp))
                output.append({**x,'split':split,'rotations':rotations,'position_controls':controls})
        f=a.output/(split+'.jsonl');f.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in output));file_hashes[f.name]=sha(f);counts[split]=dict(semantic_ids=len(sids),rows=len(output))
    (a.output/'split_ids.json').write_text(json.dumps(split_ids,indent=2)+'\n');file_hashes['split_ids.json']=sha(a.output/'split_ids.json')
    protocol=dict(frozen_at=datetime.datetime.now().astimezone().isoformat(),source='masakhane/uhura-arc-easy',source_revision='561efa15460cba0965212b77f09bf6a7f57939a9',source_split='train',languages=langs,source_counts={l:len(m) for l,m in maps.items()},common_original_ids=len(common),eligible_ids=len(eligible),excluded=excluded,cohort_semantic_ids=400,split_counts=counts,seed=seed,files_sha256=file_hashes,input_sha256=sources,max_letter_sequence_tokens=maxlen,
        scope='Project-fresh human-translated four-option science QA; original ARC training source. Not Belebele or Global-MMLU, not a substitute for frozen target benchmarks. No pretraining contamination guarantee.',
        alignment='Intersect original question IDs, same gold index and four ordered option labels; remove duplicate options, positional references, missing-visual references and direct SFT/previous-benchmark question overlap before score. Human source provenance; no new native-speaker full semantic certification.',
        primary_readouts='Original letter ACC, four cyclic content permutations, content-ID probability average, unlabeled content token-mean ACC (raw LP auxiliary). No best-prompt/permutation selection.',
        label_position_control='4 content rotations x4 cyclic display-label rotations =16 fully crossed prompts on dev/holdout. Candidates always ABCD; separately record predicted token and physical position. Noncanonical labels are auxiliary and may alter prompt naturalness.',
        p0_models=['SW_FFT_Final400','existing_SW_HA_FFT_checkpoint100'],p0_only_split='dev',holdout_rule='No holdout model scoring or teacher fitting until all pilot arms fixed and finished; all arms evaluated once, no winner-only selection.',
        teacher_rule='Only train_aux EN/SW. Same joint semantic IDs where W1 four-rotation content-average prediction is correct and strict top margin>1e-6 in BOTH languages; no dev/holdout filtering. Fixed targets T=1. All B/C/D receive same selected IDs and four rotations; teacher cost charged equally.',
        training_rule='Same W1, HA6400 cached order, seed42, AdamW1e-5, effective16, BF16/2048/nonthinking/assistant-only/no packing, original400-step cosine/warmup20 schedule stopped at100. A may reuse exact verified old100 artifact. B/C/D fixedlambda1, once per optimizer step, same auxiliary four-forward budget and labels/data. D supervised replay; B global per-language label marginal KL; C per-example content-average KL. No parameter search.',
        reporting='Freeze pilot100 Final, report HA learning and SW/EN retention, token exposures/teacher cost/time. No English-drop early stop; no automatic400 expansion. If downstream HA learning absent, prioritize task/data fit rather than longer training or new spectral methods.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:protocol[k] for k in ['cohort_semantic_ids','split_counts','common_original_ids','eligible_ids','max_letter_sequence_tokens']}))

if __name__=='__main__':main()
