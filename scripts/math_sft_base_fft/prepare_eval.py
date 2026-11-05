"""Pin all prompts before scores; official PolyMath instructions, no answer in input."""
import os, sys, json, csv, importlib.util
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from common import *
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from llamafactory.hparams import DataArguments
from llamafactory.data.template import get_template_and_fix_tokenizer
S = DATA / 'sources'
out = DATA / 'prepared-v1'
out.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('polyinstruction', S / 'PolyMath-official/instruction.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
a = DataArguments(template='qwen3', enable_thinking=False, train_on_prompt=False, cutoff_len=2048, packing=False)
t = get_template_and_fix_tokenizer(tok, a)
lock = json.loads((DATA / 'sources.lock.json').read_text())
files = {}
counts = {}
for lang, name in LANGS.items():
    records = []
    for i, x in enumerate(csv.reader((S / 'mgsm_rev2' / f'mgsm_{lang}.tsv').open(), delimiter='\t')):
        assert len(x) == 2 and number(x[1]) is not None
        records.append(dict(id='mgsm_rev2:' + str(i), benchmark='mgsm_rev2', question=x[0], answer=x[1], gold_numeric=number(x[1])))
    assert len(records) == 250
    for i, x in enumerate(rows(S / 'msvamp' / f'test_{name}.json')):
        q = x.get('m_query', x.get('question', x.get('Question', x.get('Body', ''))))
        if 'Body' in x and 'Question' in x:
            q = x['Body'] + ' ' + x['Question']
        ans = str(x.get('response', x.get('answer', x.get('Answer', ''))))
        assert q and ans and (number(ans) is not None)
        records.append(dict(id='msvamp:' + str(x.get('ID', x.get('id', i))), benchmark='msvamp', question=q, answer=ans, gold_numeric=number(ans)))
    assert len(records) == 1250
    for level in ['low', 'medium', 'high', 'top']:
        xs = pq.read_table(S / 'polymath' / lang / (level + '.parquet')).to_pylist()
        assert len(xs) == 125
        for x in xs:
            records.append(dict(id='polymath:' + level + ':' + str(x['id']), benchmark='polymath', level=level, question=x['question'], answer=x['answer']))
    for r in records:
        r['lang'] = lang
        r['prompt'] = r['question'] + '\n\n' + m.query_dic[lang] if r['benchmark'] == 'polymath' else math_prompt(lang, r['question'])
        source, _ = t.encode_multiturn(tok, [{'role': 'user', 'content': r['prompt']}, {'role': 'assistant', 'content': 'unused'}], None, None)[0]
        assert len(source) < 6144
        r['prompt_token_ids'] = source
        r['prompt_token_sha256'] = oh(source)
    p = out / f'{lang}_eval.jsonl'
    assert not p.exists()
    p.write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in records)))
    files[p.name] = sha(p)
    counts[lang] = {'mgsm_rev2': 250, 'msvamp': 1000, 'polymath': 500}
    print(lang, counts[lang], flush=True)
protocol = dict(status='FROZEN_BEFORE_ANY_NEW_SCORE', model_revision=REVISION, weights_sha256={p.name: sha(p) for p in sorted(MODEL.glob('*.safetensors'))}, tokenizer_sha256=sha(MODEL / 'tokenizer.json'), source_lock_sha256=sha(DATA / 'sources.lock.json'), files=files, counts=counts, math_prompt='uniform English instruction requesting target language; literal final-number placeholder removed; all nine Base rerun, no reuse of old different prompts', polymath_prompt='question + two newlines + official query_dic[language]; no language_control forcing', template='exact installed LLaMA-Factory qwen3, enable_thinking=False; no fewshot', generation={'temperature': 0, 'top_p': 1, 'seed': 42, 'max_model_len': 8192, 'mgsm_msvamp_max_new_tokens': 1280, 'polymath_max_new_tokens': 2048, 'max_num_seqs': 32, 'max_num_batched_tokens': 8192, 'gpu_memory_utilization': 0.8}, polymath_scoring='official first balanced boxed extraction, math_equal; medium dataset mapped to official middle; per-level ACC rounded1decimal before official weights1,2,4,8 /15', numeric_scoring='frozen common.py number last numeric and #### strict supplementary', selection='MGSM and MSVAMP average ascending rank only; PolyMath excluded; ties favor script diversity then ISO; audit training eligibility before choose', exposure='benchmark-guided exploratory language/route/epoch comparisons; ES/FR/TH and EN previously exposed. Not independent blind validation.', code_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('common.py')]})
dump(out / 'protocol.json', protocol)
