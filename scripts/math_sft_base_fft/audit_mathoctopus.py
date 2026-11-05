"""Independent per-language eligibility; no joint intersection or budget subsampling."""
from profile_paths import resolve
import json, re, collections, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from common import *
import pyarrow.parquet as pq
S = Path(resolve('MATH_SOURCE'))
REF = Path(resolve('GSM8K_TRAIN'))
canonical = {norm(x['question']): x for x in pq.read_table(REF).to_pylist()}
assert len(canonical) == 7473
name_map = {v: k for k, v in LANGS.items()}
name_map['English'] = 'en'
eval_text = {norm(x['question']) for lang in LANGS for x in rows(DATA / 'prepared-v1' / f'{lang}_eval.jsonl')}
kept = collections.defaultdict(list)
raw = collections.Counter()
reject = collections.defaultdict(collections.Counter)
seen = collections.defaultdict(set)
examples = collections.defaultdict(list)
anchors = 0
for idx, line in enumerate(S.open()):
    if not line.strip():
        continue
    x = json.loads(line)
    mat = re.search('request in (\\w+)\\.', x['prompt'])
    assert mat
    name = mat.group(1)
    lang = name_map.get(name)
    q = x['prompt'].split('### Instruction:', 1)[1].rsplit('### Response:', 1)[0].strip()
    ans = x['chosen'].strip()
    if name == 'English':
        ref = canonical.get(norm(q))
        assert ref is not None, idx
        sid = oh(norm(q))
        gold = number(ref['answer'].split('####')[-1])
        anchors += 1
    assert anchors
    if lang not in LANGS:
        continue
    raw[lang] += 1
    reason = None
    if not q or not ans:
        reason = 'empty'
    elif sid in seen[lang]:
        reason = 'duplicate_source_question'
    elif number(ans, lang) is None:
        reason = 'unparseable_final_numeric'
    elif number(ans, lang) != gold:
        reason = 'final_numeric_disagrees_with_original_gsm8k_gold'
    elif norm(q) in eval_text or norm(ref['question']) in eval_text:
        reason = 'known_exact_normalized_evaluation_question_overlap'
    if reason:
        reject[lang][reason] += 1
        if len(examples[lang]) < 20:
            examples[lang].append({'raw_row': idx, 'source_id': sid, 'reason': reason})
        continue
    seen[lang].add(sid)
    if not re.search('####\\s*[-+]?\\d', ans):
        ans += '\n#### ' + format(Decimal(gold), 'f')
    kept[lang].append({'id': sid, 'source_raw_row': idx, 'language': lang, 'question': q, 'source_question_sha256': oh(norm(ref['question'])), 'gold_numeric': gold, 'answer': ans, 'prompt': math_prompt(lang, q)})
assert anchors == 7473
out = DATA / 'mathoctopus-audit-v1'
out.mkdir(exist_ok=True)
stats = {}
for lang in LANGS:
    xs = kept[lang]
    assert len(xs) >= 1000
    p = out / f'{lang}_all_valid.jsonl'
    assert not p.exists()
    p.write_text(''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in xs)))
    stats[lang] = {'raw': raw[lang], 'valid': len(xs), 'unique_source_ids': len({x['id'] for x in xs}), 'rejections': dict(reject[lang]), 'records_sha256': sha(p), 'source_ids_sha256': oh(sorted((x['id'] for x in xs))), 'no_intersection_requirement': True, 'no_random_size_cap': True}
dump(out / 'audit.json', {'source_revision': 'dd8076b83bacd7c6ade2829317e8a500c6ffbec4', 'source_sha256': sha(S), 'canonical_gsm8k_train_sha256': sha(REF), 'anchors': anchors, 'languages': stats, 'known_overlap_rule': 'exact NFKC casefold whitespace question and canonical English source only, never same final number', 'language_semantic_and_reasoning_audit': 'NOT_RUN; final-number agreement is necessary but insufficient evidence of translation/reasoning quality', 'license': 'MathOctopus Apache-2.0; retain upstream sources. Full data not published.'})
dump(out / 'rejection_examples_private.json', examples)
print(json.dumps(stats, indent=2))
report = ROOT / 'research/math_sft_base_fft/mathoctopus_data_audit.md'
report.write_text('# MathOctopus 每语言独立合格记录\n\n来源 revision `dd8076b83bacd7c6ade2829317e8a500c6ffbec4`；原始文件 SHA256 `' + sha(S) + '`。English-led block 重建源题身份，7473 个锚点精确匹配官方 GSM8K train。每语言独立过滤，不要求九语言交集，不设任意条数上限，不扣除旧内部256题dev。\n\n|语言|原始|合格独立源题|排除|\n|---|---:|---:|---|\n' + ''.join(('|' + l + '|' + str(v['raw']) + '|' + str(v['valid']) + '|' + str(v['rejections']) + '|\n' for l, v in stats.items())) + '\n数值一致不证明翻译语义与整条推导正确；语言/推导抽样核验待做。仅排除已知精确题文/源题重叠，未声称绝对无污染。对齐身份是重建而非上游显式题ID。各语言源文件与合格记录哈希见机器审计。\n')
