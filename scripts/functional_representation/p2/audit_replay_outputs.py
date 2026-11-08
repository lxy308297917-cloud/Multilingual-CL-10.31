"""Posthoc diagnosis of the unexpected replay GSM8K decline; no rescoring."""
from pathlib import Path
import os
import collections
import hashlib
import json
import re
import statistics

R = Path(__file__).resolve().parents[3]
L = Path(os.environ['P2_LOG_ROOT'])
K = Path(os.environ['P2_CHECKPOINT_ROOT'])
O = R / 'research/functional_representation_20261109/p2'

def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def main():
    datasets, models = {}, []
    for mode in ['all_layer', 'full_feature', 'ce_replay']:
        root = L / 'evaluation' / mode / 'core'
        paths = list((root / 'results').rglob('samples_sft_v1_gsm8k_*.jsonl')); assert len(paths) == 1
        groups = collections.defaultdict(dict)
        for row in map(json.loads, paths[0].open()):
            assert row['doc_id'] not in groups[row['filter']]
            groups[row['filter']][row['doc_id']] = row
        strict, flexible = groups['strict-match'], groups['flexible-extract']
        assert len(strict) == len(flexible) == 1319 and strict.keys() == flexible.keys()
        traces = collections.defaultdict(list)
        for row in map(json.loads, (root / 'completion-metadata.jsonl').open()):
            traces[row['response_sha256']].append(row)
        joined = []
        for key, row in flexible.items():
            assert row['resps'] == strict[key]['resps']
            matching = traces[sha(row['resps'][0][0])]
            assert len(matching) == 1, 'Ambiguous or unmatched response; do not infer completion metadata'
            joined.append(matching[0])
        datasets[mode] = flexible
        four = collections.Counter((bool(strict[k]['exact_match']), bool(flexible[k]['exact_match'])) for k in strict)
        models.append(dict(model=mode, samples=1319,
            original_sample_file_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            raw_response_to_completion_metadata_uniquely_joined=1319,
            strict_correct=sum(x['exact_match'] for x in strict.values()),
            flexible_correct=sum(x['exact_match'] for x in flexible.values()),
            strict_only=four[(True, False)], flexible_only=four[(False, True)],
            final_hash_marker_present=sum('####' in x['resps'][0][0] for x in flexible.values()),
            calculator_annotation_present=sum(bool(re.search(r'<<[^>]+>>', x['resps'][0][0])) for x in flexible.values()),
            reached_1280_token_cap=sum(x['generated_tokens'] == 1280 for x in joined),
            mean_generated_tokens=statistics.mean(x['generated_tokens'] for x in joined),
            median_generated_tokens=statistics.median(x['generated_tokens'] for x in joined),
            finish_reasons=dict(collections.Counter(x['finish_reason'] for x in joined))))
    reference, replay = datasets['all_layer'], datasets['ce_replay']
    assert all(reference.keys() == x.keys() for x in datasets.values())
    for key in reference:
        assert len({tuple(x[key][field] for field in ['doc_hash', 'prompt_hash', 'target_hash']) for x in datasets.values()}) == 1
    lost = [k for k in sorted(reference) if reference[k]['exact_match'] and not replay[k]['exact_match']]
    gained = [k for k in sorted(reference) if not reference[k]['exact_match'] and replay[k]['exact_match']]
    assert lost[:5] == [2, 4, 5, 7, 17]
    old = list(map(json.loads, (K / 'ce_replay_final_v1/old_exposure.jsonl').open()))
    old_fit = {}
    for capability in ['math', 'instruction']:
        selected = [x for x in old if x['capability'] == capability]
        assert len(selected) == 1600 and [x['id'] for x in selected[:64]] == [x['id'] for x in selected[-64:]]
        old_fit[capability] = dict(exposures=1600, unique_examples=64,
            first_cycle_mean_training_ce=statistics.mean(x['old_objective'] for x in selected[:64]),
            final_cycle_mean_training_ce=statistics.mean(x['old_objective'] for x in selected[-64:]))
    report = dict(scope='POSTHOC descriptive diagnosis after replay formal core result; no tuning or score replacement',
                  models=models, same_items_and_prompt_identities=True,
                  replay_old_pool_training_fit=old_fit,
                  all_correct_replay_wrong=len(lost), all_wrong_replay_correct=len(gained),
                  manual_review_rule='First5 sorted doc_id among all-layer flexible-correct/replay flexible-wrong; selected failure cases, not an error-rate sample',
                  manual_reviewed_doc_ids=lost[:5], formal_scores_unchanged=True,
                  interpretation=['Zero replay length caps; the observed decline is not explained by truncation alone.',
                    'Manual selected cases include incorrect operations and arithmetic with a clearly wrong emitted final answer; not all errors can be attributed to answer parsing.',
                    'Calculator annotations and near-zero reused-pool training CE are descriptive observations; they do not establish a causal explanation for test decline.',
                    'This is one frozen128-example replay pool, coefficient1 and seed42; it does not show that replay generally fails.'])
    (O / 'replay_output_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (O / 'replay_output_audit.md').write_text('''# 同池CE回放的数学负结果：事后输出审查

审查在回放组正式GSM8K结果出现后添加，不改变原始生成、评分、训练或48项预定比较。三组均匹配1319题身份，并按原始生成文本的JSON哈希唯一连接真实completion元数据；不是把执行顺序当成题号。

回放组仍使用相同400步Final、目标样本流及冻结评测。`####`或`<<...>>`的出现只描述格式；不能据此确认导致退化的机制。原严格/灵活提取分数均保留，不另造“纠正后准确率”。

## 定向人工查看

取普通LoRA灵活提取正确、回放组错误的题目中，doc_id排序最前5题。以下仅发布错误摘要，原题、金答案和完整输出保留本地。这是选定的失败案例，不能用它估计所有错误类型比例。

|doc_id|实际查看到的回放错误|
|---:|---|
|2|把增值部分当作增值后的总价值，并错误处理维修支出。|
|4|把已投喂量折算到每只鸡，替代题目所求的剩余投喂量。|
|5|重复列出单项金额，但最终加总算错。|
|7|遗漏中断前的部分下载时间，最终总时长不成立。|
|17|混淆周工时、全年周数和两类工作对应的工资。|

这些案例都有明确错误的最终结果，不能仅由缺失答案标记解释。全量格式、截断、终止及提取计数见`replay_output_audit.json`。

同一旧池首轮和末轮的训练CE也记录在JSON中；它是已见样本的训练拟合，不是独立保持分数。回放是否因小池反复曝光、旧目标权重或解题风格学习而退化，目前未经过因果对照。当前负结果只适用于已冻结的回放配方；不能推广为所有回放策略无效，也不据正式测试继续挑选权重。
''')
    print(json.dumps(dict(models=3, same_items=True, lost=len(lost), gained=len(gained), scores_unchanged=True)))

if __name__ == '__main__':
    main()
