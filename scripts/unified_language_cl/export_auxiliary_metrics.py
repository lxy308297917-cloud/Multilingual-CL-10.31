"""Publishable scalar-only native diagnostics; never include parsed output counts."""
from runtime import runtime_path
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(runtime_path('project', ''))
parser = argparse.ArgumentParser()
parser.add_argument('--tracks', nargs='+', required=True)
args = parser.parse_args()
allowed = {'samples', 'primary_metric', 'score', 'hit_cap_rate', 'invalid_label_rate', 'macro_accuracy', 'macro_f1', 'strict_accuracy', 'answer_parse_rate', 'intent_accuracy', 'valid_tree_rate', 'rouge1', 'rouge2', 'rougeLsum', 'mean_generated_tokens', 'mean_reference_tokens', 'source_truncated_rate', 'language_aux_confident_n', 'language_aux_correct_n', 'language_aux_limit'}
for track in args.tracks:
    root = Path(runtime_path('results', '')) / '' / track
    out = root / 'native-auxiliary-v1'
    assert not out.exists(), out
    comparison = json.loads((root / 'comparison-test-v1/effects.json').read_text())
    assert comparison['status'] == 'COMPLETE_FIXED_FINAL_MATRIX'
    aliases = json.loads((root / 'run_aliases.json').read_text()) if (root / 'run_aliases.json').exists() else {}
    report = {'track': track, 'split': 'test', 'status': 'COMPLETE_FIXED_NATIVE_AUXILIARY_SCALARS', 'comparison_sha256': hashlib.sha256((root / 'comparison-test-v1/effects.json').read_bytes()).hexdigest(), 'limits': ['Fraction metrics retain 0..1 units, except ROUGE metrics in 0..100. Counts and token lengths are explicit.', 'Only pre-existing scorer outputs copied; no new inference, parser changes, or test-based selection.', 'Parsed prediction frequency maps deliberately excluded because their keys may contain raw outputs.'], 'models': {}}
    table = ['# ' + track + '：原生测试辅助指标', '', '保留固定测试评分器已有指标；不替代主指标或作测试集调参。数值型比例在下表转换为百分数；ROUGE原本为0—100分，长度保留token。机器可读JSON保留原始单位。', '', '| 模型 | 语言/任务 | 指标 | 值 |', '|---|---|---|---:|']
    for model in ['base', 'single_es', 'single_th', 'seq_es_th']:
        path = root / aliases.get(model + '-test-v1', model + '-test-v1')
        assert (path / 'DONE.json').exists()
        run = json.loads((path / 'run.json').read_text())
        assert comparison['sources'][model]['run_sha256'] == hashlib.sha256((path / 'run.json').read_bytes()).hexdigest()
        if track in ['xlsum', 'wikilingua']:
            assert run['primary_scorer_version'] == 'v2_explicit_multilingual_sentence_tokenizer'
        scores = json.loads((path / 'scores.json').read_text())
        groups = {}
        for key, value in scores.items():
            assert key in comparison['groups'], key
            current = {k: v for k, v in value.items() if k in allowed}
            assert all((v is None or isinstance(v, (str, int, float, bool)) for v in current.values()))
            native = comparison['groups'][key]
            scale = 1 if track in ['xlsum', 'wikilingua'] else 100
            assert native['samples'] == current['samples']
            assert abs(native['scores'][model] - scale * current['score']) < 1e-08
            groups[key] = current
            for metric, value in sorted(current.items()):
                if metric in ['samples', 'primary_metric', 'score', 'language_aux_limit']:
                    continue
                if metric.startswith('rouge'):
                    cell = f'{value:.2f} points'
                elif metric in ['mean_generated_tokens', 'mean_reference_tokens']:
                    cell = f'{value:.2f} tokens'
                elif metric in ['language_aux_confident_n', 'language_aux_correct_n']:
                    cell = str(value)
                else:
                    cell = f'{value * 100:.2f}%'
                table.append(f'| {model} | {key} | {metric} | {cell} |')
        report['models'][model] = {'scores_sha256': hashlib.sha256((path / 'scores.json').read_bytes()).hexdigest(), 'groups': groups}
    out.mkdir()
    (out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'table.md').write_text('\n'.join(table) + '\n')
    print(json.dumps({'track': track, 'status': report['status']}))
