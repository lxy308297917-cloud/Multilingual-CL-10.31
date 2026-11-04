# 统一语言顺序基线：进度与已完成结果

更新：2026-11-05T01:09:10.964126+08:00。目标进行中。

固定 ES→TH；FR只测评。各轨独立从同一锁定Qwen3-1.7B后训练版本出发，Single ES、Single TH、ES Final→TH；3200条/200步（SIB640条/40步），effective batch16，LR1e-5，assistant-only，固定Final。

| 轨道 | 数据 | Base开发 | 完成训练段数 | 正式矩阵 |
|---|---|---|---:|---|
| massive | 已冻结 | 完成 | 3/3 | 完成 |
| mathoctopus | 已冻结 | 完成 | 3/3 | 完成 |
| xnli | 已冻结 | 完成 | 3/3 | 完成 |
| mtop | 已冻结 | 完成 | 3/3 | 完成 |
| wikilingua | 已冻结 | 完成 | 3/3 | NOT_RUN/进行中 |
| xlsum | 已冻结 | 完成 | 3/3 | NOT_RUN/进行中 |
| sib200 | 已冻结 | 完成 | 3/3 | NOT_RUN/进行中 |

## 已完成的正式结果

以下是全量固定协议，非开发筛选结果；没有按分数挑checkpoint。

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


## 已完成的通用能力矩阵


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


## 当前可支持的结论

- MASSIVE有明显标签分类学习及跨语言迁移。ES→TH相对ES阶段的TH收益+4.74个百分点；ES下降0.20个百分点，配对95%区间跨零，尚不足以支持稳定旧语言遗忘。标签/输出格式掌握也是收益的可能来源，不能全归为语言能力。
- MathOctopus固定Final在MGSM-Rev2和MSVAMP上相对Base四语均下降；TH阶段虽相对W1有小幅恢复，仍低于Base。不能把这个轨道描述为“已成功学习目标语言并遗忘旧能力”。严格格式与数值正确率分列；不通过更换正式测试上的最优checkpoint美化结果。
- MASSIVE通用能力监测已完成：Base→SEQ的GSM8K宽松答案准确率73.16%→67.17%，IFEval提示级strict66.36%→44.18%；EN阅读60.89%→79.44%。已有目标分类任务收益与部分通用能力退化，不能概括为所有英语能力下降。
- MathOctopus通用能力监测已完成：GSM8K宽松答案准确率73.16%→57.39%，strict44.81%→57.24%。strict改善包含输出格式改善，不能据此声称数学能力提高。其他轨道在完成相同协议前不作通用收益/遗忘结论。
- MTOP正式原生与通用矩阵已完成：TH阶段结构EM增加36.09个百分点，IFEval prompt strict下降5.73个百分点；旧ES下降1.07的区间跨零。EN阅读同时提高。
- XNLI正式原生与通用矩阵已完成：TH阶段ACC增加6.57个百分点，GSM8K flexible下降8.57个百分点；旧ES未出现可靠下降，不能概括为所有英语能力丢失。
- 只用seed42。配对区间条件于本次训练，不覆盖训练随机性；未进行多重检验校正。

## 数据补充与限制

- WikiLingua原Drive链接404保留记录。现已取得作者参与整理的GEM官方发布并校验LFS SHA；通过原文+摘要精确匹配原始英语URL/section建立跨语言ID。3985个共同训练候选，在各语言共用3200个ID。
- GEM官方按section划分，dev/test的不同section仍共享一些原始文章。训练排除所有已映射dev/test文章URL。正式结果将明确标为唯一映射子集，并报告覆盖数；不声称完整官方test或文章级独立dev/test。
- WikiLingua评分复用锁定的XL-Sum多语言ROUGE，明确不冒充原GEM排行榜评分。MathOctopus重建ID的语义限制及XL-Sum非平行数据限制见各protocol。
- 已在任何摘要正式生成前修正上游ROUGE-Lsum的句子tokenizer传递缺失；四语所有Base/Final统一使用v2分数，原始预测保留，ROUGE1/2回归必须一致。Thai Base开发集18.54/20.58是评分修正后的XL-Sum/WikiLingua分数，不是训练收益。详见[评分修正说明](summary_scoring_correction.md)。
- 后续Same/Disjoint对照仅使用训练前协议预留的B队列。MASSIVE的A/B ID虽不重叠，仍有38个B样本的标准化泰语文本见于A；XNLI该精确文本重叠为0。MTOP也满足原预算的共同ID条件，将在优先批后补充，B中有25个输入文本见于A。保留原预留队列，不按测试分数重采样，也不把ID分离等同于语义全新。

## 运行与恢复

已完成21/21段正式训练，计数以TRAINING_DONE为准；完成的段均通过真实样本曝光、权重和固定Final核验。首批MASSIVE与MathOctopus原生及通用能力矩阵已完成。其余轨道仅将正式矩阵存在的项目标记完成。

首次测评存在CPU启动超时导致失去调度跟踪的问题，已用独立启动器修复，旧失败日志保留；另两项vLLM编译缓存失败通过独立新缓存恢复，未删除共享缓存、修改权重或改变评分。旧CPU调度器已退休。

后续只执行已授权的基线、原生任务/通用能力评测和必要预检；账号合计最多四卡，不影响他人。

- [完整任务书](task_specification.md)
- [冻结测评协议](evaluation_protocol.md)
- [四轨道阶段总表](second_batch_summary.md)
- [MTOP完整解释](mtop_results.md)
- [XNLI完整解释](xnli_results.md)
- [数学输出复核](math_output_diagnostics.md)
- [复现代码与步骤](reproduction.md)
- [Same-ID/Disjoint-ID资格与范围](p3_eligibility.md)

