"""Freeze cohorts and all prompts before scoring any new model output."""
from pathlib import Path
import argparse, collections, datetime, hashlib, json, re, unicodedata
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from lm_eval.api.model import TemplateLM
from core import permutation


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def normalized(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())


def invalid(choices):
    if any(not s.strip() for s in choices): return 'empty'
    if len(set(map(normalized, choices))) != 4: return 'duplicate_options'
    # Position-dependent options cannot be rotated while preserving semantics.
    patterns = [r'\b(all|none|both|either)\b.{0,20}\b(above|below|[a-d]\b)',
                r'\b[a-d]\s*(and|or|&|da|na|au)\s*[a-d]\b',
                r'\b(zote|yote|hakuna|vyote)\b.{0,20}\b(hapo|juu)',
                r'\b(duka|babu)\b.{0,20}\b(sama|kasa)']
    return 'position_dependent_option' if any(re.search(p, s, re.I) for p in patterns for s in choices) else None


def main():
    p = argparse.ArgumentParser()
    
    for name in ['tokenizer','knowledge-dev','reading-en','reading-sw','reading-ha','train-sw','train-ha']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); assert not a.output.exists(); a.output.mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    class Encoder:
        backend = 'causal'
        def tok_encode(self, text, **kw): return tok.encode(text, add_special_tokens=False)
    encoder = Encoder()
    def encode(text, answers):
        prompt = tok.apply_chat_template([{'role':'user','content':text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        pairs = [TemplateLM._encode_pair(encoder, prompt, s) for s in answers]
        assert all(c and t and len(c)+len(t)<=2048 for c,t in pairs)
        return pairs
    sources = {}; rows = []; exclusions = []; counts = collections.Counter()
    raw = a.knowledge_dev
    langs = ['en','sw','ha']
    maps = {}
    for l in langs:
        f = raw / (l+'-dev.parquet'); sources[str(f)] = sha(f)
        values = pq.read_table(f).to_pylist(); maps[l] = {x['sample_id']:x for x in values}
        assert len(values)==len(maps[l])==285
    assert all(set(m)==set(maps['en']) for m in maps.values())
    grouped = collections.defaultdict(list)
    for sid, v in maps['en'].items(): grouped[v['subject']].append(sid)
    seed = 'answer-bias-calibration-20261103-v1'
    for subject, ids in sorted(grouped.items()):
        ordered = sorted(ids, key=lambda s:hashlib.sha256((seed+'|'+s).encode()).hexdigest())
        for n, sid in enumerate(ordered):
            split = 'calibration' if n==0 else 'diagnostic'
            triple=[]
            for l in langs:
                v=maps[l][sid]; options=[v['option_'+c] for c in 'abcd']
                assert v['answer']==maps['en'][sid]['answer'] and v['subject']==subject
                triple.append(dict(id=sid,cluster=sid,task='knowledge_dev',language=l,
                                   question=v['question'].strip(), passage='', choices=options,
                                   gold='ABCD'.index(v['answer']),subject=subject,split=split))
            reasons=[invalid(x['choices']) for x in triple]
            if any(reasons): exclusions.append(dict(task='knowledge_dev',id=sid,reasons=reasons,split=split));continue
            rows.extend(triple)
    reading_paths={'en':a.reading_en,
                   'sw':a.reading_sw,
                   'ha':a.reading_ha}
    selected=[]
    rm={}
    for l,f in reading_paths.items():
        sources[str(f)]=sha(f);rm[l]={(v['link'],str(v['question_number'])):v for v in map(json.loads,f.open())}
    reading_groups=collections.defaultdict(list)
    for key in rm['en']: reading_groups[key[0]].append(key)
    reading_seed='ha-learning-reading-development-v1-20261103'
    for link in sorted(reading_groups,key=lambda s:hashlib.sha256((reading_seed+'|'+s).encode()).hexdigest()):
        keys=sorted(reading_groups[link])
        if len(selected)+len(keys)<=180:selected+=keys
        if len(selected)==180:break
    assert len(selected)==180
    for link,qnum in selected:
        triple=[]
        for l in langs:
            v=rm[l][link,str(qnum)];assert v['correct_answer_num']==rm['en'][link,str(qnum)]['correct_answer_num']
            triple.append(dict(id=link+'|'+str(qnum),cluster=link,task='reading_exposed',language=l,
                               question=v['question'].strip(),passage=v['flores_passage'],choices=[v['mc_answer'+str(i)] for i in range(1,5)],gold=int(v['correct_answer_num'])-1,subject='reading',split='diagnostic'))
        reasons=[invalid(x['choices']) for x in triple]
        if any(reasons): exclusions.append(dict(task='reading_exposed',id=link+'|'+str(qnum),reasons=reasons));continue
        rows.extend(triple)
    # Necessary, limited exact question / answer checks against already frozen SFT.
    trainfiles=[a.train_ha,
                a.train_sw]
    train_questions=set(); checked=[]
    for f in trainfiles:
        if f.exists():
            checked.append(str(f)); sources[str(f)]=sha(f)
            for x in map(json.loads,f.open()): train_questions.add(normalized(x.get('instruction',x.get('prompt',''))))
    overlaps={(x['task'],x['id']) for x in rows if normalized(x['question']) in train_questions}
    rows=[x for x in rows if (x['task'],x['id']) not in overlaps]
    encoded=[]
    for x in rows:
        base=('P: '+x['passage']+'\n' if x['passage'] else '')+'Q: '+x['question']+'\n'
        rotations=[]
        for r in range(4):
            order=permutation(r); opts=[x['choices'][i] for i in order]
            letter=base+'\n'.join(c+': '+s for c,s in zip('ABCD',opts))+'\nAnswer:'
            content=base+'Options:\n'+'\n'.join('- '+s for s in opts)+'\nAnswer with the complete text of the correct option.\nAnswer:'
            rotations.append(dict(rotation=r,order=order,gold_position=order.index(x['gold']),
                                  letter_pairs=encode(letter,list('ABCD')),content_pairs=encode(content,opts)))
        encoded.append({**x,'rotations':rotations}); counts[x['task']+'|'+x['split']+'|'+x['language']]+=1
    cf=[]
    # Fixed content-free prompts; no diagnostic questions or answers are used.
    for task in ['knowledge_dev','reading_exposed']:
        for placeholder in ['N/A','[empty]','']:
            base=('P: '+placeholder+'\n' if task=='reading_exposed' else '')+'Q: '+placeholder+'\n'
            text=base+'\n'.join(c+': '+placeholder for c in 'ABCD')+'\nAnswer:'
            cf.append(dict(task=task,placeholder=placeholder,letter_pairs=encode(text,list('ABCD'))))
    f=a.output/'items.jsonl'; f.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in encoded))
    c=a.output/'content_free.json';c.write_text(json.dumps(cf))
    protocol=dict(frozen_at=datetime.datetime.now().astimezone().isoformat(),seed=seed,
                  counts=dict(counts),exclusions=exclusions,exact_train_question_overlaps_removed=sorted(overlaps),
                  train_sources_checked=checked,source_sha256=sources,files_sha256={f.name:sha(f),c.name:sha(c)},
                  model_selection=False,training=False,new_independent_confirmation=False,
                  exposure='Global-MMLU official dev285 historically exposed as prompts/pilots; hash-split one per subject for calibration, remaining four for diagnostic. Calibration never uses formal test. Belebele fixed prior180 exposed diagnostic only; no fitting or prompt selection on these test-origin items.',
                  translation='Belebele official professional parallel translations; Global-MMLU full translations share IDs/gold but are not independently semantically certified, so cross-language knowledge analysis exploratory.',
                  rotations='four cyclic rotations 0123,1230,2301,3012; no best permutation selection',
                  arithmetic='FP32 exports, SDPA, no TF32, eval, right pad, float32 lm_head/logsoftmax, batch4; exact causal continuation scores',
                  content_scoring='unlabelled option list; exact official answer strings, no generated aliases; sum and mean token LP both retained. Primary content comparison mean token LP. Changed task prompt is a limitation, not pure semantic ground truth.',
                  priors='PriDe eq7 per calibration ID then arithmetic mean, per language; transferred from knowledge dev to reading (explicit cross-task). Content-free prior mean of3 fixed no-content prompts per task, shared across languages. No labels, strength=1, floor1e-12.',
                  gate='Run calibration comparisons only for a model/task/language where diagnostic rotation-averaged predicted-position maxshare>=0.40 and content-unanimity<=0.80. Gate frozen before outputs; no threshold tuning.',
                  random_control='all23 nonidentity label-prior permutations, same calibration inputs and magnitude; report their complete distribution, never select best',
                  analyses='paired 2000 bootstrap seed42 by passage for reading, semantic ID knowledge; exploratory unadjusted95%; same-ID crosslanguage and stage contrasts. No causal localization.',
                  optional_full24='NOT_RUN: bounded4rotation primary design; no adaptive expansion')
    (a.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'counts':dict(counts),'excluded':len(exclusions),'overlaps':len(overlaps),'protocol_sha256':sha(a.output/'protocol.json')}))

if __name__=='__main__':main()
