# XNLI：目标任务学习与通用能力变化

全量原生测试与统一通用能力测评均完成。各语言原生test为5010题，主指标为三分类ACC。数据标签顺序已与固定Parquet的官方ClassLabel元数据核对：entailment、neutral、contradiction。

每段3200例、200 optimizer steps、有效batch16、seed42、LR1e-5、AdamW、BF16、assistant-only、non-thinking、固定Final。SEQ复用ES阶段，再从该权重进行TH训练，优化器和scheduler重置。Single TH为200步；SEQ逻辑累计400步，不能当作相同总计算量。

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


## 如何理解原生结果

- TH阶段收益：+6.57 [+5.35, +7.86]个百分点。
- SEQ相对Single TH：+0.28 [-0.64, +1.24]个百分点；区间跨零，不能称为可靠的额外TH收益。
- ES遗忘：-0.46 [-1.22, +0.26]个百分点；未看到可靠的旧ES任务下降。
- ES训练后未见TH的迁移为+13.15 [+11.76, +14.52]个百分点。任务和标签读出学习也参与这一收益，不等于从零学会泰语。

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


## 当前结论

- 已观察到TH任务提升同时发生部分通用任务退化：同一TH阶段，GSM8K flexible下降8.57个百分点，IFEval prompt strict下降4.07个百分点；配对95%区间均不跨零。
- 相对Base，SEQ的GSM8K flexible为42.00%（下降31.16个百分点），IFEval prompt strict为57.30%（下降9.06个百分点）。与此同时EN Belebele为81.67%，比Base提高20.78个百分点。不能概括为所有英语能力都下降。
- strict格式GSM8K分数另列，不能用其下降幅度替代flexible数值准确率的变化。这里只确认冻结任务协议上的行为变化，不单凭分数宣布内部数学知识被擦除。
- 当前结果不证明语言顺序本身导致这些通用退化：SEQ还包含更多窄任务训练，Same-ID也重复跨语言语义内容。预留B对照用于帮助解释内容与队列难度；仍不是纯语言因果实验。
- 仅seed42；配对区间条件于当前固定模型，不涵盖训练随机性，未进行多重检验校正。未用这些测试成绩改变学习率、数据、预算或选择checkpoint。

全部21段主训练的固定预算、相同Base初始化与实际TH样本顺序已核验；其他轨道仍继续正式测评。整个研究目标尚未完成。
