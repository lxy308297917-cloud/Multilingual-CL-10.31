"""Bounded CPU language-ID diagnostics; never changes answers, scores or training.

The same ID-hash-selected items are used for each checkpoint. A language detector
on mathematical text is an auxiliary proxy, not a human correctness judgment.
"""
from profile_paths import resolve
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv, gzip, json, sys, hashlib
from pathlib import Path
from common import *
import fasttext

def main():
    result = ROOT / 'results/math_sft_base_fft'
    assert (result / 'routeA-analysis-v1/DONE.json').exists()
    out = result / 'output-language-audit-v1'
    assert not out.exists()
    out.mkdir()
    model_path = Path(resolve('GLOTLID'))
    model = fasttext.load_model(str(model_path))
    expected = {'sw': 'swh', 'bn': 'ben', 'th': 'tha', 'ja': 'jpn', 'zh': 'cmn', 'ru': 'rus', 'fr': 'fra', 'de': 'deu', 'es': 'spa'}
    base = [v for i in [1, 2, 3] for v in rows(result / f'base-census-worker{i}-v1/predictions.jsonl')]
    summaries = []
    measured = []
    selection = {}
    for lang in LANGS:
        for ep in [0, 1, 2] if lang in ['sw', 'bn', 'ja'] else [0]:
            xs = [x for x in base if x['lang'] == lang] if ep == 0 else rows(result / f'A/{lang}/epoch{ep}/math/predictions.jsonl')
            for benchmark in ['mgsm_rev2', 'msvamp', 'polymath']:
                source = [x for x in xs if x['benchmark'] == benchmark]
                if benchmark == 'polymath':
                    sample = [x for level in ['low', 'medium', 'high', 'top'] for x in sorted([v for v in source if v['level'] == level], key=lambda v: oh([42, 'math-output-language', v['id']]))[:16]]
                else:
                    sample = sorted(source, key=lambda v: oh([42, 'math-output-language', v['id']]))[:64]
                assert len(sample) == 64
                ids = [x['id'] for x in sample]
                key = lang + ':' + benchmark
                if ep == 0:
                    selection[key] = ids
                else:
                    assert selection[key] == ids, 'Paired language audit samples changed'
                records = []
                for x in sample:
                    text = x['output'].replace('\n', ' ')[:6000]
                    rec = {'language': lang, 'epoch': ep, 'benchmark': benchmark, 'level': x.get('level'), 'item_id_sha256': hashlib.sha256(str(x['id']).encode()).hexdigest(), 'prompt_token_sha256': x['prompt_token_sha256'], 'measured': sum((c.isalpha() for c in text)) >= 50, 'generated_tokens': x['generated_tokens'], 'hit_length_cap': x['finish_reason'] == 'length'}
                    if rec['measured']:
                        labels, probs = model.predict(text, k=3)
                        pred = labels[0].replace('__label__', '')
                        rec.update(top1=pred, top1_probability=float(probs[0]), predicted_target=pred.split('_')[0] == expected[lang], predicted_english=pred.split('_')[0] == 'eng', high_confidence=float(probs[0]) >= 0.8)
                    records.append(rec)
                vals = [x for x in records if x['measured']]
                summaries.append({'language': lang, 'epoch': ep, 'benchmark': benchmark, 'selected': len(records), 'measured': len(vals), 'short_or_math_dominated_not_measured': len(records) - len(vals), 'top1_predicted_target': sum((x['predicted_target'] for x in vals)), 'top1_predicted_english': sum((x['predicted_english'] for x in vals)), 'top1_other_language': sum((not x['predicted_target'] and (not x['predicted_english']) for x in vals)), 'high_confidence_predicted_target': sum((x['predicted_target'] and x['high_confidence'] for x in vals)), 'selected_id_list_sha256': oh(ids)})
                measured.extend(records)
    with (out / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    with (out / 'per_item_diagnostics.jsonl.gz').open('wb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as z:
            z.write(''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in measured)).encode())
    dump(out / 'summary.json', summaries)
    dump(out / 'DONE.json', {'status': 'COMPLETE_AUXILIARY_LANGUAGE_ID_DIAGNOSTIC', 'model': 'GlotLID model_v3.bin', 'model_sha256': sha(model_path), 'seed': 42, 'sample_rule': 'Each math task selects the same 64 items by fixed ID hash; PolyMath stratifies 16 per difficulty. Up to 6000 output characters; fewer than 50 alphabetic characters not measured.', 'roles': 15, 'role_task_groups': 45, 'selected_outputs': len(measured), 'used_GPU': False, 'not_used_for_scoring_selection_or_data_filtering': True, 'limitations': 'Automatic language ID on formula-rich text is not human-verified output-language correctness. English technical/math fragments and script confusion can affect labels; short outputs remain explicitly unmeasured. No generated text or answer is exported.'})
    table = '|语言|节点|任务|抽样数|可识别文本数|识别为目标语言|识别为英语|其他语言|\n|---|---|---|---:|---:|---:|---:|---:|\n' + ''.join((f"|{x['language']}|{('Base' if x['epoch'] == 0 else 'E' + str(x['epoch']))}|{x['benchmark']}|{x['selected']}|{x['measured']}|{x['top1_predicted_target']}|{x['top1_predicted_english']}|{x['top1_other_language']}|\n" for x in summaries))
    (ROOT / 'research/math_sft_base_fft/output_language_diagnostics.md').write_text('# 输出语言辅助诊断\n\n' + table + '\n每组固定64题；PolyMath按四难度各16题分层。三模型节点严格使用相同题目。GlotLID v3只对前6000字符且至少50个字母的输出进行识别，过短或主要为公式的输出保留未测，不计为语言错误。上表数字是自动识别计数，不能当作人工确认的语言正确率，也不改变任何正式benchmark分母或分数。全部结果仅作语言/格式诊断，不用于数据过滤、选checkpoint或路线门槛。\n')
    print(json.dumps({'status': 'COMPLETE_AUXILIARY_LANGUAGE_ID_DIAGNOSTIC', 'role_task_groups': len(summaries), 'selected_outputs': len(measured), 'used_GPU': False}))
if __name__ == '__main__':
    main()
