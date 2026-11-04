# MTOP：固定预算语言顺序微调结果

原生全量测试与统一通用能力测评均已完成。主指标为组合式解耦语义树的结构完全匹配率（Tree EM），不是仅预测根意图的分类ACC。固定Final，不因英语下降早停，也不从测试分数挑checkpoint。

## 配方与实际执行

官方Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；每段3200例、200 optimizer steps、有效batch16、seed42、LR1e-5、AdamW、BF16、assistant-only、non-thinking、长度2048、不packing。SEQ复用同轨ES Final，仅新增TH段；逻辑上累计400步/6400次样本曝光，不能把其与Single TH的200步当作相同总训练量。

全项目21段主训练均已完成实际预算核验，且7轨道Single TH与SEQ TH的两个rank实际输入顺序逐条相同。此处MTOP使用冻结的prepared-v2目标形式。

## 原生任务：结构EM（%）

| 语言 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| es | 2998 | 0.13 | 66.18 | 45.26 | 65.11 |
| th | 2765 | 0.00 | 12.91 | 45.68 | 49.01 |
| fr | 3193 | 0.00 | 51.55 | 42.15 | 53.05 |
| en | 4386 | 0.75 | 52.94 | 36.53 | 55.75 |

关键配对变化及95%区间（百分点）：

- TH阶段收益：+36.09 [+34.14, +37.97]。
- SEQ相对直接Single TH：+3.33 [+2.03, +4.63]。
- ES遗忘（ES Final减SEQ Final）：+1.07 [-0.27, +2.40]；区间跨零，不能称为稳定的旧西语遗忘。

TH合法树比例在ES阶段已经达到98.88%，SEQ为99.93%；同期TH结构EM从12.91%到49.01%。这部分阶段收益不能仅用“学会输出合法括号格式”解释。绝对Base→微调收益仍同时包含任务、schema和输出形式学习，不能全部归为一般语言能力。

## 通用能力：完整固定Final对照

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


## 当前结论

- 这是一条已观察到目标任务增益与部分通用能力下降的FFT基线：TH阶段增加结构EM约36.09个百分点，同时IFEval prompt strict下降5.73个百分点、instruction strict下降4.44个百分点，配对区间均不跨零。
- 不是所有英语能力都下降：同一TH阶段EN Belebele提高6.00个百分点；GSM8K flexible变化−0.23个百分点，区间跨零。相对Base的GSM8K flexible下降3.71个百分点，不能把strict下降20.47个百分点全部当作同等数学能力损失。
- 本轮不能以它证明明显旧ES任务遗忘或内部英语知识擦除；目前观察到的是任务相关的功能变化。
- 当前Same-ID配置重复了跨语言内容与任务结构；SEQ相对Single TH还包含额外的ES训练。后续预留B队列的对照用于帮助解释内容重复与队列难度，不把差值直接称为纯语言切换因果效应。
- 仅seed42，区间条件于本次模型与测试题目；不覆盖训练随机性，未做多重检验校正。不同语言的原生测试集合大小不同，不冒充跨语言逐题配对。

## 来源

- [原生配对结果](../../results/unified_language_cl/mtop/comparison-test-v1/effects.json)
- [通用能力配对结果](../../results/unified_language_cl/mtop/retention-comparison-v1/effects.json)
- [Same-ID/Disjoint-ID资格与预先冻结计划](p3_eligibility.md)
