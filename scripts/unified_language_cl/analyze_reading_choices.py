"""Reconstruct fixed EN Belebele ACC and option distributions from stored likelihoods.

No generation, option permutation, parser modification or model selection.
"""
from runtime import runtime_path
import collections
import json
import math
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, runtime_path('project', 'scripts/unified_language_cl'))
from common import ROOT, sha, rows, dump
from aggregate_retention import directory
TRACKS = ['massive', 'mathoctopus', 'mtop', 'xnli', 'xlsum', 'wikilingua', 'sib200']
LABELS = ['A', 'B', 'C', 'D']
OUT = Path(runtime_path('results', '')) / 'en-reading-choice-audit-v1'
assert not OUT.exists()
records, reference, checked_base = ({}, None, None)
table = ['# EN Belebele：选项似然与标签偏置复核', '', '冻结阅读协议是0-shot multiple_choice：比较A/B/C/D续写token的log-likelihood，主指标ACC。它不是自由生成答案后用正则提取字母，因此下降不能解释为单纯“输出了摘要而解析失败”。标签先验、任务读出与阅读表征变化仍可能共同影响选项似然。', '', '复核全部22份不同模型权重的同一900题，Base共享一次；由已保存的四选项似然逐题重建ACC并核对正式分数，不改变提示、选项顺序或评分。不据这些测试材料调整训练。', '', '| 模型 | n | 原ACC% | 重建ACC% | 预测A% | 预测B% | 预测C% | 预测D% | 最集中选项及比例 |', '|---|---:|---:|---:|---:|---:|---:|---:|---|']
for track in TRACKS:
    formal = json.loads((Path(runtime_path('results', '')) / f'{track}/retention-comparison-v1/effects.json').read_text())
    for arm in ['base', 'single_es', 'single_th', 'seq_es_th']:
        key = 'base' if arm == 'base' else track + ':' + arm
        if key in records:
            assert records[key]['accuracy_percent'] == formal['groups']['reading:acc:none']['scores'][arm]
            continue
        folder = directory(track, arm)
        assert (folder / 'DONE.json').exists()
        files = list((folder / 'en_reading/results').rglob('samples_*.jsonl'))
        assert len(files) == 1
        sample_file = files[0]
        data = rows(sample_file)
        assert len(data) == len({r['doc_id'] for r in data}) == 900
        identities = {r['doc_id']: (r['doc_hash'], r['prompt_hash'], r['target_hash']) for r in data}
        if reference is None:
            reference = identities
        assert identities == reference
        confusion = [[0] * 4 for _ in range(4)]
        correct = 0
        for r in data:
            assert r['filter'] == 'none'
            ll = [float(value[0]) for value in r['filtered_resps']]
            assert len(ll) == 4 and all((math.isfinite(value) for value in ll))
            gold = int(r['target'])
            assert gold == int(r['doc']['correct_answer_num']) - 1 and 0 <= gold < 4
            predicted = max(range(4), key=lambda j: ll[j])
            hit = predicted == gold
            assert float(hit) == float(r['acc']), ('Original scorer disagreement', key, r['doc_id'])
            correct += hit
            confusion[gold][predicted] += 1
        n = len(data)
        accuracy = 100 * correct / n
        published = formal['groups']['reading:acc:none']['scores'][arm]
        assert abs(accuracy - published) < 1e-09
        gold_counts = [sum(row) for row in confusion]
        predicted_counts = [sum((row[j] for row in confusion)) for j in range(4)]
        majority = max(range(4), key=lambda j: predicted_counts[j])
        records[key] = {'samples': n, 'accuracy_percent': accuracy, 'published_accuracy_percent': published, 'choice_order': LABELS, 'gold_counts': dict(zip(LABELS, gold_counts)), 'predicted_counts': dict(zip(LABELS, predicted_counts)), 'confusion_gold_rows_predicted_columns': confusion, 'most_predicted_label': LABELS[majority], 'most_predicted_share': predicted_counts[majority] / n, 'original_acc_matches_every_sample': True, 'source_samples_sha256': sha(sample_file), 'completion_receipt_sha256': sha(folder / 'DONE.json')}
        table.append(f'| {key} | {n} | {published:.2f} | {accuracy:.2f} | ' + ' | '.join((f'{100 * value / n:.2f}' for value in predicted_counts)) + f' | {LABELS[majority]} {100 * predicted_counts[majority] / n:.2f}% |')
assert len(records) == 22
table += ['', '## 解释边界', '', '- 选项分布集中提示答案token先验或任务读出可能参与退化，但仅凭分布不能识别其因果贡献，也不能区分语言理解、标签校准与其他参数变化。', '- 本轮没有置换选项、修改提示或再生成；这些属于额外干预，标记NOT_RUN。没有用正式测试选择更好的解析规则。', '- 原始四选项分数完整保留在本地，公开只提供混淆矩阵和标量统计，不再分发原文或逐题响应。GSM8K和IFEval属于生成式评测，需要分别分析，不能把本结论套到所有任务。', '- 这是已完成benchmark的事后描述性错误分析；目标收益与遗忘仍以原协议正式成绩和已有配对区间为准。']
OUT.mkdir()
dump(OUT / 'audit.json', {'status': 'ALL_22_FIXED_READING_MODELS_RECONSTRUCTED_EXACTLY', 'models': records, 'same_900_docs_prompts_targets': True, 'new_generation': False, 'test_used_for_changes': False, 'protocol': '0-shot four-label log-likelihood ACC, not generated-letter parsing'})
(OUT / 'report.md').write_text('\n'.join(table) + '\n')
print(json.dumps({'status': 'ALL_22_FIXED_READING_MODELS_RECONSTRUCTED_EXACTLY', 'unique_models': len(records), 'questions_per_model': 900}))
