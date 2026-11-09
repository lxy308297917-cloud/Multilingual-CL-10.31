# XQuAD补充结果

2026-11-09T23:41:37.328130+08:00

AR、RU各1190题、240段文章。SW不受官方支持，标记UNSUPPORTED。XQuAD保留为补充，不加入固定5+5平均。主评分使用官方SQuAD v1.1英文规范化、参考答案最大EM/F1，输出原文直接评分，不按测试成绩选答案提取方式。

| 语言 | 模型 | EM% | F1% |
|---|---|---:|---:|
| ar | ar_Base | 0.08 | 26.46 |
| ar | ar_historical_lora | 49.33 | 66.71 |
| ru | ru_Base | — | — |
| ru | ru_historical_lora | — | — |
| ar | ar_all | — | — |
| ar | ar_function | — | — |
| ru | ru_all | — | — |
| ru | ru_full | — | — |
| ru | ru_random | — | — |
| ru | ru_function | — | — |

输出格式诊断见format_diagnostics.json。带答案标签、额外单位或完整句的输出会受到严格EM/F1惩罚，不能将低EM直接等同于不会阅读。Markdown符号本身会被官方ASCII标点规则去除，额外词语仍保留。诊断不改变原始评分或提示。

—为尚未完成；四个尚未训练且按用户要求暂缓的模型另见deferred_models.json，未用其他模型冒充。按文章分簇配对区间见paired_changes.csv，仅反映固定模型样本不确定性。没有新增SQuAD训练，不与论文中SQuAD专门训练后的抽取式模型直接排名。
