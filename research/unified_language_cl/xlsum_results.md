# XL-Sum：目标摘要学习与跨任务退化

全量四语言原生测试与统一通用能力测评已完成。主指标为ROUGE-Lsum F1分，不是ACC。原生test：ES4763、TH826、FR1086、EN11535。不同语言数据不是同题翻译，不能将语言间绝对分数差解释为纯语言难度。

每段3200例、200 optimizer steps、有效batch16、seed42、LR1e-5、AdamW、BF16、长度2048、assistant-only、non-thinking、固定Final；摘要microbatch1/accumulation8、双卡ZeRO3。SEQ复用ES Final并重置优化器与调度器，再完成200步TH；不因通用能力下降而早停，不用test挑checkpoint。

## 评分版本

下面全部使用预先冻结的v2多语言句子分词修正：对保留的同一份生成输出统一重新评分，没有重新生成。作者实现的ROUGE-Lsum句子分词未传语言tokenizer；修正仅补齐这一调用，R1/R2逐样本回归一致。所有Base/Final统一处理，保留原始评分与预测。不能将旧、新评分差当作训练收益。详见[评分修正说明](summary_scoring_correction.md)。

# xlsum：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| xlsum | en | 11535 | 10.40 | 23.81 | 20.66 | 23.03 |
| xlsum | es | 4763 | 11.88 | 20.99 | 19.44 | 21.15 |
| xlsum | fr | 1086 | 15.63 | 24.14 | 22.85 | 24.04 |
| xlsum | th | 826 | 18.62 | 16.53 | 25.60 | 25.67 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:xlsum | ES阶段迁移 | +13.41 [+13.23, +13.60] |
| en:xlsum | TH阶段变化 | -0.78 [-0.93, -0.62] |
| es:xlsum | G_ES | +9.11 [+8.82, +9.41] |
| es:xlsum | F_ES | -0.16 [-0.39, +0.06] |
| fr:xlsum | ES阶段迁移 | +8.51 [+7.86, +9.20] |
| fr:xlsum | TH阶段变化 | -0.10 [-0.66, +0.49] |
| th:xlsum | Single TH收益 | +6.99 [+6.42, +7.55] |
| th:xlsum | G_TH | +9.14 [+8.55, +9.76] |
| th:xlsum | T_TH | -2.09 [-2.52, -1.66] |
| th:xlsum | SEQ Final−Single TH Final | +0.07 [-0.32, +0.46] |
| th:xlsum | 阶段收益差（非因果） | +2.16 [+1.59, +2.75] |

## 原生学习的含义

- Single ES的西语摘要从11.88升至20.99；Single TH的泰语摘要从18.62升至25.60，存在匹配任务学习。
- ES训练后泰语摘要降至16.53，比Base低2.09分，95%区间[-2.52,-1.66]。TH阶段恢复至25.67，阶段增益9.14[8.55,9.76]。
- SEQ相对Single TH仅高0.07[-0.32,0.46]，没有可靠的额外终点优势。较大的阶段收益部分来自较低的ES后泰语起点，不能只用阶段增益宣布正向前向迁移。
- 西语20.99→21.15，遗忘-0.16[-0.39,0.06]，没有可靠旧西语摘要遗忘。英语摘要23.81→23.03，下降0.78分[0.62,0.93]；这是同任务未训练英语上的变化。

# xlsum：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 39.42 | 52.08 | 35.78 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 11.14 | 1.67 | 0.38 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 23.50 | 40.05 | 25.66 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 11.46 | 26.25 | 12.94 |
| reading:acc:none | 900 | 60.89 | 22.78 | 24.33 | 22.78 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -3.64 [-6.29, -0.83] | -37.38 [-40.33, -34.42] |
| gsm8k:exact_match:strict-match | -10.77 [-12.51, -9.17] | -44.43 [-47.08, -41.77] |
| ifeval:inst_level_strict_acc:none | +2.16 [+0.00, +4.29] | -49.28 [-53.28, -45.26] |
| ifeval:prompt_level_strict_acc:none | +1.48 [-0.55, +3.51] | -53.42 [-58.23, -48.42] |
| reading:acc:none | +0.00 [-0.56, +0.56] | -38.11 [-41.56, -34.33] |

## 通用能力与机制限制

- 相对Base，Final的GSM8K flexible从73.16%降至35.78%，IFEval prompt strict从66.36%降至12.94%，EN Belebele从60.89%降至22.78%。与原生摘要改善同时出现广泛的跨任务行为退化。
- 退化在ES第一段已经很大；不能将所有下降归因于TH切换。TH阶段的GSM8K flexible再降3.64个百分点[-6.29,-0.83]，但阅读无进一步变化，IFEval prompt变化区间跨零。
- 摘要输出风格专门化、任务分布与读出失配仍是可能解释。当前分数并不证明通用知识已从参数中被擦除，也不是纯语言切换因果实验。正式测试不用于修改提示、解码、评分或训练预算。
- 该轨道非严格平行，不开展Same-ID/Disjoint-ID伪对照；其他合格轨道的内容对照继续按原计划执行。
- 配对区间为2000次bootstrap，条件于seed42当前固定模型；不覆盖训练随机性，未做多重检验校正。所有成绩按各自单位保留，不与ACC混合求平均。

## 可追溯结果

- [四语言测试矩阵](../../results/unified_language_cl/xlsum/comparison-test-v1/table.md)
- [测试配对效应与输入来源哈希](../../results/unified_language_cl/xlsum/comparison-test-v1/effects.json)
- [统一通用能力表](../../results/unified_language_cl/xlsum/retention-comparison-v1/table.md)
- [完整48组开发读出诊断](../../results/unified_language_cl/classification_readout_complete_v2/table.md)：只覆盖分类轨道，不替代XL-Sum测评。

其他轨道与预定P3对照尚未全部完成，整个目标继续进行。

## 输出长度与语言复核

[完整16组长度、截断和语言识别表](../../results/unified_language_cl/xlsum/output-audit-v1/table.md)复用已有生成记录：

- 泰语平均生成token为Base346.3、Single ES410.1、Single TH140.8、SEQ137.8，参考平均126.6。Single ES达到生成上限的比例为60.90%，远高于Base10.90%及Final0.85%。
- 因此ES后泰语摘要下降、TH后恢复伴随明显的长度控制变化；目前不能把全部ROUGE改善解释为新增语义知识。没有通过追加生成或改变上限来选择更好成绩。
- 泰语源文章截断率59.08%，四个模型相同。正式结果描述固定上下文预算下的摘要表现，不代表完整原文条件，跨语言绝对分数也不宜直接比较。
- 泰语四组自动语言识别均为100%被识别成泰语；这只降低了“大量切换成另一语言”的可能性，不是人工准确性核验，不说明摘要内容正确或忠实。
- 其他语言也同时报告识别覆盖率、正确/全部与正确/有置信三项，避免低置信输出被隐藏。摘要质量结论仍以固定原生ROUGE为主。
