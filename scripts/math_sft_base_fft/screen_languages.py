"""Select only after all nine complete; no PolyMath in selection ranking."""
import collections, json, csv, math
from common import *
R = ROOT / 'results/math_sft_base_fft'
out = R / 'base-language-screening-v1'
assert not out.exists()
out.mkdir()
assert (R / 'base-census-polymath-v1/DONE.json').exists()
allrows = []
sources = {}
for worker in [1, 2, 3]:
    d = R / f'base-census-worker{worker}-v1'
    assert (d / 'GENERATION_DONE.json').exists()
    run = json.loads((d / 'run.json').read_text())
    assert run['protocol_sha256'] == sha(DATA / 'prepared-v1/protocol.json')
    assert run['weights_sha256'] == json.loads((DATA / 'prepared-v1/protocol.json').read_text())['weights_sha256']
    allrows += rows(d / 'predictions.jsonl')
    sources[d.name] = {'run_sha256': sha(d / 'run.json'), 'predictions_sha256': sha(d / 'predictions.jsonl')}
audit = json.loads((DATA / 'mathoctopus-audit-v1/audit.json').read_text())
vals = []
for lang in LANGS:
    record = {'language': lang, 'eligible_training_samples': audit['languages'][lang]['valid']}
    for task, n in [('mgsm_rev2', 250), ('msvamp', 1000)]:
        xs = [r for r in allrows if r['lang'] == lang and r['benchmark'] == task]
        assert len(xs) == n and len({r['id'] for r in xs}) == n
        record[task + '_correct'] = sum((x['correct'] for x in xs))
        record[task + '_n'] = n
        record[task + '_acc'] = sum((x['correct'] for x in xs)) / n
        record[task + '_parse_rate'] = sum((x['parsed'] is not None for x in xs)) / n
        record[task + '_hit_cap_rate'] = sum((x['finish_reason'] == 'length' for x in xs)) / n
        record[task + '_strict_acc'] = sum((x['strict_correct'] for x in xs)) / n
        record[task + '_mean_output_tokens'] = sum((x['generated_tokens'] for x in xs)) / n
    poly = json.loads((R / 'base-census-polymath-v1' / f'{lang}-score/scores.json').read_text())
    record['polymath_DW_ACC_percent'] = poly[lang + ':DW-ACC']['official_percent']
    for level in ['low', 'medium', 'high', 'top']:
        record['polymath_' + level + '_acc'] = poly[lang + ':' + level]['accuracy']
    vals.append(record)
for r in vals:
    for task in ['mgsm_rev2', 'msvamp']:
        score = r[task + '_acc']
        r[task + '_rank'] = 1 + sum((x[task + '_acc'] < score for x in vals)) + (sum((x[task + '_acc'] == score for x in vals)) - 1) / 2
    r['average_rank'] = (r['mgsm_rev2_rank'] + r['msvamp_rank']) / 2
    r['parse_investigation_flag'] = any((r[t + '_parse_rate'] < 0.9 for t in ['mgsm_rev2', 'msvamp']))
    r['floor_investigation_flag'] = any((r[t + '_acc'] <= 0.02 for t in ['mgsm_rev2', 'msvamp']))
script = {'sw': 'Latin', 'bn': 'Bengali', 'th': 'Thai', 'ja': 'Japanese', 'zh': 'Han', 'ru': 'Cyrillic', 'fr': 'Latin', 'de': 'Latin', 'es': 'Latin'}
pool = [r for r in vals if r['eligible_training_samples'] >= 1000]
selected = []
used = set()
while len(selected) < 3:
    low = min((r['average_rank'] for r in pool))
    bucket = [r for r in pool if r['average_rank'] == low]
    r = min(bucket, key=lambda r: (script[r['language']] in used, r['language']))
    selected.append(r)
    pool.remove(r)
    used.add(script[r['language']])
flags = [r['language'] for r in selected if r['parse_investigation_flag'] or r['floor_investigation_flag']]
result = {'status': 'NEEDS_PARSE_OR_FLOOR_REVIEW' if flags else 'COMPLETE', 'selected_languages': [r['language'] for r in selected], 'review_flags': flags, 'all_languages': sorted(vals, key=lambda r: (r['average_rank'], r['language'])), 'selection_protocol_sha256': sha(DATA / 'prepared-v1/protocol.json'), 'source_runs': sources, 'selection_scope': 'BENCHMARK_GUIDED_EXPLORATORY_NOT_BLIND_VALIDATION', 'polymath_used_for_selection': False}
dump(out / 'selection.json', result)
with (out / 'all_languages.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(vals[0]))
    w.writeheader()
    w.writerows(result['all_languages'])
header = '|语言|MGSM正确/250|MGSM ACC%|MSVAMP正确/1000|MSVAMP ACC%|平均低分名次|PolyMath DW-ACC%（不参与选语言）|合格训练题|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
table = header + ''.join((f"|{r['language']}|{r['mgsm_rev2_correct']}|{100 * r['mgsm_rev2_acc']:.2f}|{r['msvamp_correct']}|{100 * r['msvamp_acc']:.2f}|{r['average_rank']:.2f}|{r['polymath_DW_ACC_percent']:.1f}|{r['eligible_training_samples']}|\n" for r in result['all_languages']))
(ROOT / 'research/math_sft_base_fft/base_language_screening.md').write_text('# 九语言 Base 完整普查\n\n' + table + '\n全部九语言使用同一英文指令结构、目标语言回答要求，重新测量；旧不同提示结果未混用。已暴露正式题用于探索性选语言，不能宣称盲测。PolyMath评分复用官方表达式等价，未参与排序。解析/截断/长度和分难度原始聚合见CSV。\n')
(ROOT / 'research/math_sft_base_fft/selected_languages.md').write_text('# 三语言选择\n\n状态：' + result['status'] + '\n\n按训练前规则选定：' + ', '.join(result['selected_languages']) + '。两项Base ACC分别从低到高排名，同分平均名次，再取两名次平均；仅完全同名次时优先脚本多样性和ISO字母序。每语言全部合格训练题均>=1000。\n\n需复核标记：' + str(flags) + '。选定后不因训练分数改变语言；不是三语言共同交集采样。\n\n' + table)
print(json.dumps(result, indent=2))
