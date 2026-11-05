"""Expose existing summary length/language diagnostics, without new inference."""
from runtime import runtime_path
import argparse
import json
import hashlib
from pathlib import Path
ROOT = Path(runtime_path('project', ''))
parser = argparse.ArgumentParser()
parser.add_argument('--track', choices=['xlsum', 'wikilingua'], required=True)
args = parser.parse_args()
root = Path(runtime_path('results', '')) / '' / args.track
out = root / 'output-audit-v1'
assert not out.exists()
aliases = json.loads((root / 'run_aliases.json').read_text())
table = ['# ' + args.track + '：生成长度与语言诊断', '', '以下复用冻结生成时记录的辅助指标；不新增生成、不重新判语言、不改变ROUGE或测试配置。语言识别为langdetect1.0.9自动估计，不是人工语言正确率。低置信/过短文本记为未测，不从分母中悄悄剔除；同时报告覆盖率和两种分母。', '', '| 模型 | 语言 | n | ROUGE-Lsum | 平均生成token | 平均参考token | 达到长度上限% | 源截断% | 识别覆盖% | 识别正确/全部% | 识别正确/有置信% |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
result = {}
for model in ['base', 'single_es', 'single_th', 'seq_es_th']:
    path = root / aliases[model + '-test-v1']
    assert (path / 'DONE.json').exists()
    run = json.loads((path / 'run.json').read_text())
    assert run['primary_scorer_version'] == 'v2_explicit_multilingual_sentence_tokenizer'
    values = json.loads((path / 'scores.json').read_text())
    result[model] = {'scores_sha256': hashlib.sha256((path / 'scores.json').read_bytes()).hexdigest(), 'groups': {}}
    for key, v in sorted(values.items()):
        lang = key.split(':')[0]
        n, known, correct = (v['samples'], v['language_aux_confident_n'], v['language_aux_correct_n'])
        assert 0 <= correct <= known <= n
        record = {k: v[k] for k in ['samples', 'score', 'mean_generated_tokens', 'mean_reference_tokens', 'hit_cap_rate', 'source_truncated_rate', 'language_aux_confident_n', 'language_aux_correct_n', 'language_aux_limit']}
        record.update(language_detection_coverage=known / n, detected_correct_over_all=correct / n, detected_correct_over_confident=correct / known if known else None)
        result[model]['groups'][key] = record
        confident = f'{100 * correct / known:.2f}' if known else 'UNMEASURED'
        table.append(f"| {model} | {lang} | {n} | {v['score']:.2f} | {v['mean_generated_tokens']:.1f} | {v['mean_reference_tokens']:.1f} | {100 * v['hit_cap_rate']:.2f} | {100 * v['source_truncated_rate']:.2f} | {100 * known / n:.2f} | {100 * correct / n:.2f} | {confident} |")
table += ['', '参考token使用固定Qwen tokenizer；生成长度包含协议内实际生成token。长度与语言指标用于解释表现，不能代替ROUGE或支持方法去留。多语言识别存在短文本、混语与类别偏差；这些数值不应称作人工核验的语言正确率。']
out.mkdir()
(out / 'effects.json').write_text(json.dumps({'status': 'COMPLETE_FIXED_OUTPUT_AUDIT', 'track': args.track, 'split': 'test', 'new_inference': False, 'models': result}, ensure_ascii=False, indent=2) + '\n')
(out / 'table.md').write_text('\n'.join(table) + '\n')
print(out)
