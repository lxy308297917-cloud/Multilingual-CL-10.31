# SIB-200：小样本主题学习与通用能力变化

原生四语言test及统一通用能力测评均完成。每语言test204题，主指标为生成式7类ACC，同时保留Macro-F1与非法标签率。每段640例/40 optimizer steps，有效batch16、seed42、LR1e-5、AdamW、BF16、长度2048、assistant-only、non-thinking、固定Final。没有重复小数据来凑3200例。SEQ复用ES Final，重置优化器与scheduler，再做40步TH；累计80步，不等于Single TH总预算。

# sib200：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| sib200 | en | 204 | 46.08 | 87.75 | 87.75 | 90.20 |
| sib200 | es | 204 | 50.49 | 85.78 | 85.78 | 88.73 |
| sib200 | fr | 204 | 41.67 | 86.27 | 86.27 | 89.71 |
| sib200 | th | 204 | 15.20 | 83.82 | 84.80 | 87.25 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:sib200 | ES阶段迁移 | +41.67 [+34.31, +49.02] |
| en:sib200 | TH阶段变化 | +2.45 [-0.49, +5.88] |
| es:sib200 | G_ES | +35.29 [+27.45, +43.14] |
| es:sib200 | F_ES | -2.94 [-7.35, +1.47] |
| fr:sib200 | ES阶段迁移 | +44.61 [+36.76, +52.45] |
| fr:sib200 | TH阶段变化 | +3.43 [+0.00, +6.86] |
| th:sib200 | Single TH收益 | +69.61 [+62.75, +76.47] |
| th:sib200 | G_TH | +3.43 [+0.00, +6.86] |
| th:sib200 | T_TH | +68.63 [+61.76, +75.00] |
| th:sib200 | SEQ Final−Single TH Final | +2.45 [-1.47, +6.86] |
| th:sib200 | 阶段收益差（非因果） | -66.18 [-73.53, -58.33] |

## 完整测试辅助指标

| 模型 | 语言 | Macro-F1% | 非法标签率% |
|---|---|---:|---:|
| base | en | 35.17 | 0.00 |
| base | es | 44.17 | 0.00 |
| base | fr | 31.15 | 0.00 |
| base | th | 22.03 | 80.88 |
| single_es | en | 87.52 | 0.49 |
| single_es | es | 85.57 | 0.49 |
| single_es | fr | 85.88 | 0.49 |
| single_es | th | 82.58 | 0.98 |
| single_th | en | 86.74 | 0.00 |
| single_th | es | 84.79 | 0.00 |
| single_th | fr | 85.64 | 0.00 |
| single_th | th | 83.11 | 0.00 |
| seq_es_th | en | 89.57 | 0.49 |
| seq_es_th | es | 88.23 | 0.49 |
| seq_es_th | fr | 88.89 | 0.00 |
| seq_es_th | th | 85.24 | 0.00 |

## 读出限制

[冻结开发64题复核](../../results/unified_language_cl/classification_readout_complete_v2/table.md)在同题、相同提示上比较生成与合法候选token log-prob：泰语Base生成14.06%、候选48.44%，非法标签率78.12%；Final生成85.94%、候选87.50%，非法标签率0%。

这说明原始泰语生成地板受到标签读出影响，但不能用“全是格式修复”解释候选成绩也提高的现象。候选材料透露任务和标签，不能单独证明从零获得泰语能力；正式成绩仍为全量204题生成ACC。

第一阶段对未训练泰语已有很大迁移（正式15.20→83.82），第二阶段增益只有3.43个百分点，95%区间下界为0.00。SEQ相对Single TH高2.45[-1.47,6.86]，没有可靠终点优势。旧西语85.78→88.73，未见可靠旧语言下降。不能仅凭巨大的Base→Final差忽略任务读出与跨语迁移。

# sib200：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 73.09 | 70.43 | 69.60 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 46.93 | 61.49 | 22.06 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 70.62 | 69.78 | 60.43 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 61.00 | 60.07 | 49.72 |
| reading:acc:none | 900 | 60.89 | 67.00 | 78.56 | 69.33 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -3.49 [-5.61, -1.36] | -3.56 [-5.84, -1.29] |
| gsm8k:exact_match:strict-match | -24.87 [-28.13, -21.53] | -22.74 [-26.16, -19.56] |
| ifeval:inst_level_strict_acc:none | -10.19 [-13.27, -6.84] | -14.51 [-18.04, -10.84] |
| ifeval:prompt_level_strict_acc:none | -11.28 [-15.53, -7.02] | -16.64 [-21.07, -12.20] |
| reading:acc:none | +2.33 [-0.67, +5.11] | +8.44 [+4.78, +12.00] |

## 当前结论

- 相同TH阶段，GSM8K flexible下降3.49个百分点[-5.61,-1.36]，IFEval prompt strict下降11.28[-15.53,-7.02]；出现部分通用行为损失。
- EN阅读相对Base改善8.44[4.78,12.00]；TH阶段变化区间跨零，不能概括为英语全面退化。
- strict GSM8K大幅下降另列，主数学解释使用flexible数值准确率。任务格式专门化与真实能力变化尚未完全分开；不能据此断言内部知识擦除。
- 该轨道共同train ID701，不够两组各640的Disjoint-ID预算；不事后缩预算做有利对照，不重复样本冒充独立数据。
- 原生test仅204题，各区间为条件于seed42模型的2000次配对bootstrap，未做多重检验校正，不覆盖训练随机性。没有根据正式test改变训练、解码、评分或选择checkpoint。

[完整原生效应及来源哈希](../../results/unified_language_cl/sib200/comparison-test-v1/effects.json)；[完整通用效应](../../results/unified_language_cl/sib200/retention-comparison-v1/effects.json)。WikiLingua和P3对照尚未完成，整个研究目标仍继续。
