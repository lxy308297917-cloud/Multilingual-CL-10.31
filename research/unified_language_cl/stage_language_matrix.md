# 四语言阶段矩阵与统一通用任务

所有结果均为预定Final，不使用正式test早停、选checkpoint或调参。每语言阶段3200例/200步，SIB-200为640例/40步；有效batch16、seed42、LR1e-5、AdamW、BF16、assistant-only、non-thinking、长度2048、无packing。SEQ共享同一个Single ES Final，TH阶段重置优化器与调度器；Single TH和SEQ的TH实际输入/标签哈希顺序相同。SEQ逻辑累计预算是Single TH的两倍，不能当相同总计算量。

原生分类ACC、MTOP结构Tree EM、数学最终数字ACC以百分数报告，摘要为ROUGE-Lsum F1分；不同任务不混合平均。所有95%区间条件于seed42固定模型，不涵盖训练随机性，未做多重检验校正。区间跨零不代表等价。

XL-Sum文章非平行：只在同语言内将相同测试文章与模型配对。WikiLingua为官方划分的唯一映射非空子集，按英语文章聚类bootstrap；不能称全量官方test或dev/test文章级独立。摘要全部用同一v2多语言句子分词修正，不混用旧版本分数。

以下原生与通用矩阵原样嵌入；各语言实际样本量、Single TH对照和逐效应区间完整保留。通用任务不拼成单一EN遗忘分数。

## massive

# massive：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| massive | en | 2974 | 37.26 | 85.88 | 84.67 | 86.42 |
| massive | es | 2974 | 24.48 | 84.03 | 79.66 | 83.83 |
| massive | fr | 2974 | 14.66 | 83.46 | 81.81 | 84.90 |
| massive | th | 2974 | 4.94 | 76.60 | 80.33 | 81.34 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:massive | ES阶段迁移 | +48.62 [+46.74, +50.44] |
| en:massive | TH阶段变化 | +0.54 [-0.20, +1.31] |
| es:massive | G_ES | +59.55 [+57.70, +61.43] |
| es:massive | F_ES | +0.20 [-0.67, +1.08] |
| fr:massive | ES阶段迁移 | +68.80 [+67.11, +70.48] |
| fr:massive | TH阶段变化 | +1.45 [+0.57, +2.35] |
| th:massive | Single TH收益 | +75.39 [+73.87, +76.87] |
| th:massive | G_TH | +4.74 [+3.63, +5.88] |
| th:massive | T_TH | +71.65 [+70.11, +73.27] |
| th:massive | SEQ Final−Single TH Final | +1.01 [+0.07, +1.98] |
| th:massive | 阶段收益差（非因果） | -70.65 [-72.49, -68.83] |

# massive：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 69.29 | 53.90 | 67.17 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 9.40 | 1.06 | 5.00 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 63.79 | 57.55 | 56.24 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 51.76 | 45.84 | 44.18 |
| reading:acc:none | 900 | 60.89 | 76.89 | 80.56 | 79.44 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -2.12 [-4.32, -0.08] | -5.99 [-8.49, -3.34] |
| gsm8k:exact_match:strict-match | -4.40 [-6.07, -2.73] | -39.80 [-42.76, -36.92] |
| ifeval:inst_level_strict_acc:none | -7.55 [-10.66, -4.47] | -18.71 [-22.65, -14.62] |
| ifeval:prompt_level_strict_acc:none | -7.58 [-11.46, -3.70] | -22.18 [-26.99, -17.19] |
| reading:acc:none | +2.56 [+0.78, +4.44] | +18.56 [+15.00, +22.11] |

辅助指标（Intent ACC、Macro-F1、有效输出、ROUGE1/2等）见[完整标量表](../../results/unified_language_cl/massive/native-auxiliary-v1/table.md)。

## mathoctopus

# mathoctopus：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| mgsm_rev2 | en | 250 | 80.80 | 63.60 | 63.60 | 61.60 |
| msvamp | en | 1000 | 90.60 | 65.10 | 60.30 | 59.90 |
| mgsm_rev2 | es | 250 | 74.80 | 56.00 | 52.80 | 52.80 |
| msvamp | es | 1000 | 82.60 | 64.60 | 59.30 | 63.80 |
| mgsm_rev2 | fr | 250 | 71.60 | 54.00 | 55.60 | 51.20 |
| msvamp | fr | 1000 | 83.40 | 60.90 | 58.10 | 59.20 |
| mgsm_rev2 | th | 250 | 58.00 | 43.20 | 44.40 | 45.60 |
| msvamp | th | 1000 | 75.50 | 56.80 | 56.50 | 59.10 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:mgsm_rev2 | ES阶段迁移 | -17.20 [-24.00, -10.80] |
| en:mgsm_rev2 | TH阶段变化 | -2.00 [-8.00, +4.00] |
| en:msvamp | ES阶段迁移 | -25.50 [-28.50, -22.60] |
| en:msvamp | TH阶段变化 | -5.20 [-7.80, -2.60] |
| es:mgsm_rev2 | G_ES | -18.80 [-24.80, -12.40] |
| es:mgsm_rev2 | F_ES | +3.20 [-2.80, +9.20] |
| es:msvamp | G_ES | -18.00 [-21.10, -15.00] |
| es:msvamp | F_ES | +0.80 [-1.80, +3.30] |
| fr:mgsm_rev2 | ES阶段迁移 | -17.60 [-24.80, -10.40] |
| fr:mgsm_rev2 | TH阶段变化 | -2.80 [-8.41, +3.60] |
| fr:msvamp | ES阶段迁移 | -22.50 [-25.60, -19.50] |
| fr:msvamp | TH阶段变化 | -1.70 [-4.40, +1.00] |
| th:mgsm_rev2 | Single TH收益 | -13.60 [-20.00, -7.60] |
| th:mgsm_rev2 | G_TH | +2.40 [-4.00, +8.40] |
| th:mgsm_rev2 | T_TH | -14.80 [-21.20, -8.00] |
| th:mgsm_rev2 | SEQ Final−Single TH Final | +1.20 [-4.00, +6.40] |
| th:mgsm_rev2 | 阶段收益差（非因果） | +16.00 [+8.00, +24.40] |
| th:msvamp | Single TH收益 | -19.00 [-22.20, -15.70] |
| th:msvamp | G_TH | +2.30 [-0.50, +5.20] |
| th:msvamp | T_TH | -18.70 [-21.60, -15.50] |
| th:msvamp | SEQ Final−Single TH Final | +2.60 [+0.20, +5.20] |
| th:msvamp | 阶段收益差（非因果） | +21.30 [+17.20, +25.40] |

# mathoctopus：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 62.62 | 59.97 | 57.39 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 62.47 | 60.05 | 57.24 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 53.48 | 48.80 | 49.64 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 39.74 | 34.94 | 35.49 |
| reading:acc:none | 900 | 60.89 | 42.67 | 27.44 | 45.33 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -5.23 [-7.81, -2.58] | -15.77 [-18.65, -12.81] |
| gsm8k:exact_match:strict-match | -5.23 [-7.81, -2.58] | +12.43 [+9.17, +15.70] |
| ifeval:inst_level_strict_acc:none | -3.84 [-6.73, -1.07] | -25.30 [-29.19, -21.33] |
| ifeval:prompt_level_strict_acc:none | -4.25 [-7.58, -0.74] | -30.87 [-35.67, -25.88] |
| reading:acc:none | +2.67 [+1.22, +4.11] | -15.56 [-19.22, -11.67] |

辅助指标（Intent ACC、Macro-F1、有效输出、ROUGE1/2等）见[完整标量表](../../results/unified_language_cl/mathoctopus/native-auxiliary-v1/table.md)。

## xnli

# xnli：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| xnli | en | 5010 | 44.85 | 86.37 | 85.11 | 86.69 |
| xnli | es | 5010 | 48.74 | 82.08 | 81.22 | 82.53 |
| xnli | fr | 5010 | 56.89 | 80.24 | 80.00 | 82.04 |
| xnli | th | 5010 | 55.25 | 68.40 | 74.69 | 74.97 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:xnli | ES阶段迁移 | +41.52 [+39.91, +43.13] |
| en:xnli | TH阶段变化 | +0.32 [-0.52, +1.18] |
| es:xnli | G_ES | +33.33 [+31.75, +34.96] |
| es:xnli | F_ES | -0.46 [-1.22, +0.26] |
| fr:xnli | ES阶段迁移 | +23.35 [+21.83, +24.93] |
| fr:xnli | TH阶段变化 | +1.80 [+0.92, +2.69] |
| th:xnli | Single TH收益 | +19.44 [+17.94, +20.92] |
| th:xnli | G_TH | +6.57 [+5.35, +7.86] |
| th:xnli | T_TH | +13.15 [+11.76, +14.52] |
| th:xnli | SEQ Final−Single TH Final | +0.28 [-0.64, +1.24] |
| th:xnli | 阶段收益差（非因果） | -12.87 [-14.53, -11.14] |

# xnli：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 50.57 | 66.94 | 42.00 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 6.07 | 17.06 | 4.47 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 70.86 | 68.71 | 68.47 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 61.37 | 58.23 | 57.30 |
| reading:acc:none | 900 | 60.89 | 82.00 | 80.44 | 81.67 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -8.57 [-10.92, -6.37] | -31.16 [-34.19, -28.13] |
| gsm8k:exact_match:strict-match | -1.59 [-3.11, -0.15] | -40.33 [-43.14, -37.68] |
| ifeval:inst_level_strict_acc:none | -2.40 [-5.10, +0.23] | -6.47 [-9.55, -3.66] |
| ifeval:prompt_level_strict_acc:none | -4.07 [-7.39, -1.10] | -9.06 [-13.31, -5.18] |
| reading:acc:none | -0.33 [-2.00, +1.33] | +20.78 [+17.56, +23.89] |

辅助指标（Intent ACC、Macro-F1、有效输出、ROUGE1/2等）见[完整标量表](../../results/unified_language_cl/xnli/native-auxiliary-v1/table.md)。

## mtop

# mtop：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| mtop | en | 4386 | 0.75 | 52.94 | 36.53 | 55.75 |
| mtop | es | 2998 | 0.13 | 66.18 | 45.26 | 65.11 |
| mtop | fr | 3193 | 0.00 | 51.55 | 42.15 | 53.05 |
| mtop | th | 2765 | 0.00 | 12.91 | 45.68 | 49.01 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:mtop | ES阶段迁移 | +52.19 [+50.71, +53.63] |
| en:mtop | TH阶段变化 | +2.80 [+1.60, +3.94] |
| es:mtop | G_ES | +66.04 [+64.41, +67.75] |
| es:mtop | F_ES | +1.07 [-0.27, +2.40] |
| fr:mtop | ES阶段迁移 | +51.55 [+49.89, +53.30] |
| fr:mtop | TH阶段变化 | +1.50 [+0.19, +2.82] |
| th:mtop | Single TH收益 | +45.68 [+43.80, +47.59] |
| th:mtop | G_TH | +36.09 [+34.14, +37.97] |
| th:mtop | T_TH | +12.91 [+11.72, +14.25] |
| th:mtop | SEQ Final−Single TH Final | +3.33 [+2.03, +4.63] |
| th:mtop | 阶段收益差（非因果） | -9.58 [-11.50, -7.67] |

# mtop：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 69.67 | 66.79 | 69.45 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 40.18 | 27.52 | 24.34 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 69.66 | 65.95 | 65.23 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 60.26 | 55.64 | 54.53 |
| reading:acc:none | 900 | 60.89 | 65.00 | 60.33 | 71.00 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -0.23 [-2.35, +2.05] | -3.71 [-6.29, -1.36] |
| gsm8k:exact_match:strict-match | -15.85 [-18.50, -13.04] | -20.47 [-23.73, -16.98] |
| ifeval:inst_level_strict_acc:none | -4.44 [-7.38, -1.67] | -9.71 [-13.44, -6.22] |
| ifeval:prompt_level_strict_acc:none | -5.73 [-9.43, -2.40] | -11.83 [-16.64, -7.39] |
| reading:acc:none | +6.00 [+4.00, +8.00] | +10.11 [+7.33, +12.89] |

辅助指标（Intent ACC、Macro-F1、有效输出、ROUGE1/2等）见[完整标量表](../../results/unified_language_cl/mtop/native-auxiliary-v1/table.md)。

## xlsum

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

辅助指标（Intent ACC、Macro-F1、有效输出、ROUGE1/2等）见[完整标量表](../../results/unified_language_cl/xlsum/native-auxiliary-v1/table.md)。

## wikilingua

# wikilingua：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| wikilingua | en | 27004 | 18.25 | 33.05 | 31.91 | 32.89 |
| wikilingua | es | 21164 | 17.34 | 30.14 | 27.29 | 28.85 |
| wikilingua | fr | 12128 | 17.73 | 27.29 | 27.18 | 27.30 |
| wikilingua | th | 2723 | 21.01 | 16.12 | 31.52 | 32.99 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:wikilingua | ES阶段迁移 | +14.80 [+14.63, +14.98] |
| en:wikilingua | TH阶段变化 | -0.17 [-0.29, -0.05] |
| es:wikilingua | G_ES | +12.81 [+12.61, +13.00] |
| es:wikilingua | F_ES | +1.30 [+1.16, +1.43] |
| fr:wikilingua | ES阶段迁移 | +9.56 [+9.29, +9.82] |
| fr:wikilingua | TH阶段变化 | +0.01 [-0.18, +0.20] |
| th:wikilingua | Single TH收益 | +10.51 [+9.90, +11.15] |
| th:wikilingua | G_TH | +16.86 [+16.23, +17.51] |
| th:wikilingua | T_TH | -4.89 [-5.29, -4.49] |
| th:wikilingua | SEQ Final−Single TH Final | +1.46 [+1.03, +1.91] |
| th:wikilingua | 阶段收益差（非因果） | +6.35 [+5.77, +6.96] |

# wikilingua：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 56.71 | 58.00 | 35.94 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 6.14 | 0.00 | 0.00 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 49.52 | 46.76 | 37.65 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 36.60 | 32.35 | 25.14 |
| reading:acc:none | 900 | 60.89 | 36.44 | 75.11 | 28.89 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -20.77 [-23.65, -17.89] | -37.23 [-40.03, -34.12] |
| gsm8k:exact_match:strict-match | -6.14 [-7.43, -4.93] | -44.81 [-47.46, -42.15] |
| ifeval:inst_level_strict_acc:none | -11.87 [-15.06, -8.39] | -37.29 [-41.61, -32.97] |
| ifeval:prompt_level_strict_acc:none | -11.46 [-15.16, -7.58] | -41.22 [-46.40, -36.22] |
| reading:acc:none | -7.56 [-11.00, -4.22] | -32.00 [-36.33, -27.78] |

辅助指标（Intent ACC、Macro-F1、有效输出、ROUGE1/2等）见[完整标量表](../../results/unified_language_cl/wikilingua/native-auxiliary-v1/table.md)。

## sib200

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

辅助指标（Intent ACC、Macro-F1、有效输出、ROUGE1/2等）见[完整标量表](../../results/unified_language_cl/sib200/native-auxiliary-v1/table.md)。

## 如何解释这些比较

- 新语言阶段改善并不自动意味着顺序学习优于直接Single TH；必须同时看SEQ−Single TH及更长累计预算。负迁移应结合ES后起点和TH正对照，不能只比较最终绝对分数。
- 分类任务的标签/读出学习、MTOP树输出、摘要长度与格式都可能贡献分数变化。见[同题开发读出诊断](../../results/unified_language_cl/classification_readout_complete_v2/table.md)、[数学输出复核](math_output_diagnostics.md)及各摘要输出诊断。
- 部分通用任务下降不等于全部英语能力下降，也不单凭行为分数证明内部知识擦除。不同语言同任务保持与跨任务通用保持分别报告。
- 主实验Same-ID重复跨语言内容，不能分离内容复习与语言切换；P3的预留Disjoint-ID还引入内容难度差异，不是纯语言因果实验。对照尚未完成时，不以本表宣布研究目标完成。

