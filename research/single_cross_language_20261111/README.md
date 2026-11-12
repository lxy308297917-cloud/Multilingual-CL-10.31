# 三个独立普通LoRA模型的跨语言迁移矩阵

更新时间：2026-11-11T22:54:53.902359+08:00

**状态：全部60个主分与90组配对区间完成。**

## 目的与预先固定的对比

检验SW（斯瓦希里语）、AR（阿拉伯语）、RU（俄语）single训练的收益是否只出现在对应语言。行是测试语言，列是独立训练语言；所有模型各自从同一底座初始化，没有顺序训练。对同一语言的同一批题，比较Base、SW single、AR single、RU single；另列相对Base涨点和相对对应语言single的差值。RA按前文确认视为AR。

single指从同一原模型独立微调一个目标语言。LoRA（低秩适配）训练附加矩阵；词元（token）是分词器切分后的计算单位。底座Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c；三个adapter均为micro2×累积8、6400例/400步的普通LoRA Final400，seed42，不混入micro1历史模型或表示保护方法。LoRA rank16/alpha32/dropout0.05，36层q/k/v/o/gate/up/down，长度2048，assistant-only，学习率1e-4 cosine，warmup0.05。SW/AR/RU输入词元2373891/2265373/2520204，不等词元预算。

每语训练混合阅读1920、原生摘要1280、指令摘要640、知识640、常识640、EN→目标语640、目标语→EN640。包含双语文本，训练语言不是唯一变化因素；模型已经过底座指令训练，不能将进一步SFT概括为从零学指令。模型和训练数据身份见plan.json。

## 输入提示、评分与计算

复用此前5+5固定协议，不改变题面、候选顺序、聊天模板、few-shot、解码或评分。完整训练输入示例及真实渲染提示在[既有方法记录](../sequential_lora_20261110/protocol.json)，框架配置、提示模板和数据来源在[协议](../sequential_lora_20261110/protocol.json)。配对时只更换模型，同一题号与输入身份哈希必须相同，不跨语言直接将不同题配对。

Belebele各900题/0-shot、XNLI各5010题/0-shot、Global-MMLU各14042题57学科/5-shot dev、SIB各204题/0-shot。采用实际lm-evaluation-harness0.4.14.dev0，上游d6de81643928d653435c431bae19945d41d32520，本地离线数据和任务提示适配。Belebele固定P/Q/A-D/Answer与bare A-D候选；XNLI固定relation说明与bare A-C；知识用学科dev五例；SIB固定7个英文标签和字符长度归一化主分。Qwen3 non-thinking聊天模板。

XL-Sum为非选择题对照：SW987、AR4689、RU7780篇，固定贪心vLLM生成，逐篇ROUGE-Lsum均值。SW使用rouge-score0.1.2默认词元与English Punkt分句，AR/RU使用XL-Sum afcd803上游多语言评分加既有native-tokenizer传递修正，不声称未经修改的官方端到端。摘要绝对分数不跨语言排名。

ACC正确率为正确题数/总题数×100；归一化SIB按冻结候选长度归一化似然取最大。除摘要为ROUGE分，其他均为百分数，差值为百分点。配对95%区间使用2000次bootstrap，seed42；阅读按篇章、XNLI按premise聚类、摘要按文章，其余按题。区间只反映固定模型的测试样本变化，不是训练seed不确定性，无多重比较修正。

## Belebele：完整模型×测试语言矩阵

本表每行使用该测试语言的全部固定正式样本，范围见上节。Base是原模型；SW/AR/RU single各指只经过相应语言一次完整训练的模型，对角线是对应语言训练。比较同一行；各语言题目/分词不能视为只改语言的严格因果对照。

| 测试语言 | Base | SW single | AR single | RU single |
|---|---|---|---|---|
| SW | 27.33 | 36.33 | 41.00 | 40.00 |
| AR | 46.22 | 77.89 | 80.67 | 79.33 |
| RU | 55.56 | 82.44 | 82.33 | 82.00 |

| 测试语言 | 训练模型 | 相对Base | 相对对应single |
|---|---|---|---|
| SW | SW_single | +9.00 | +0.00 |
| AR | SW_single | +31.67 | -2.78 |
| RU | SW_single | +26.89 | +0.44 |
| SW | AR_single | +13.67 | +4.67 |
| AR | AR_single | +34.44 | +0.00 |
| RU | AR_single | +26.78 | +0.33 |
| SW | RU_single | +12.67 | +3.67 |
| AR | RU_single | +33.11 | -1.33 |
| RU | RU_single | +26.44 | +0.00 |

SW：对应single较Base变化+9.00；其他语言训练后，AR_single较Base+13.67，较对应single+4.67、RU_single较Base+12.67，较对应single+3.67。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

AR：对应single较Base变化+34.44；其他语言训练后，SW_single较Base+31.67，较对应single-2.78、RU_single较Base+33.11，较对应single-1.33。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

RU：对应single较Base变化+26.44；其他语言训练后，SW_single较Base+26.89，较对应single+0.44、AR_single较Base+26.78，较对应single+0.33。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

## XNLI：完整模型×测试语言矩阵

本表每行使用该测试语言的全部固定正式样本，范围见上节。Base是原模型；SW/AR/RU single各指只经过相应语言一次完整训练的模型，对角线是对应语言训练。比较同一行；各语言题目/分词不能视为只改语言的严格因果对照。

| 测试语言 | Base | SW single | AR single | RU single |
|---|---|---|---|---|
| SW | 33.19 | 42.22 | 33.43 | 34.69 |
| AR | 33.81 | 33.85 | 36.93 | 54.63 |
| RU | 37.35 | 42.67 | 36.89 | 60.14 |

| 测试语言 | 训练模型 | 相对Base | 相对对应single |
|---|---|---|---|
| SW | SW_single | +9.02 | +0.00 |
| AR | SW_single | +0.04 | -3.07 |
| RU | SW_single | +5.33 | -17.47 |
| SW | AR_single | +0.24 | -8.78 |
| AR | AR_single | +3.11 | +0.00 |
| RU | AR_single | -0.46 | -23.25 |
| SW | RU_single | +1.50 | -7.52 |
| AR | RU_single | +20.82 | +17.70 |
| RU | RU_single | +22.79 | +0.00 |

SW：对应single较Base变化+9.02；其他语言训练后，AR_single较Base+0.24，较对应single-8.78、RU_single较Base+1.50，较对应single-7.52。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

AR：对应single较Base变化+3.11；其他语言训练后，SW_single较Base+0.04，较对应single-3.07、RU_single较Base+20.82，较对应single+17.70。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

RU：对应single较Base变化+22.79；其他语言训练后，SW_single较Base+5.33，较对应single-17.47、AR_single较Base-0.46，较对应single-23.25。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

## XL-Sum：完整模型×测试语言矩阵

本表每行使用该测试语言的全部固定正式样本，范围见上节。Base是原模型；SW/AR/RU single各指只经过相应语言一次完整训练的模型，对角线是对应语言训练。比较同一行；各语言题目/分词不能视为只改语言的严格因果对照。

| 测试语言 | Base | SW single | AR single | RU single |
|---|---|---|---|---|
| SW | 13.71 | 20.18 | 8.83 | 13.09 |
| AR | 9.21 | 18.99 | 20.83 | 19.42 |
| RU | 10.07 | 17.06 | 18.02 | 18.86 |

| 测试语言 | 训练模型 | 相对Base | 相对对应single |
|---|---|---|---|
| SW | SW_single | +6.47 | +0.00 |
| AR | SW_single | +9.77 | -1.84 |
| RU | SW_single | +6.99 | -1.80 |
| SW | AR_single | -4.88 | -11.35 |
| AR | AR_single | +11.62 | +0.00 |
| RU | AR_single | +7.95 | -0.83 |
| SW | RU_single | -0.62 | -7.09 |
| AR | RU_single | +10.21 | -1.41 |
| RU | RU_single | +8.78 | +0.00 |

SW：对应single较Base变化+6.47；其他语言训练后，AR_single较Base-4.88，较对应single-11.35、RU_single较Base-0.62，较对应single-7.09。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

AR：对应single较Base变化+11.62；其他语言训练后，SW_single较Base+9.77，较对应single-1.84、RU_single较Base+10.21，较对应single-1.41。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

RU：对应single较Base变化+8.78；其他语言训练后，SW_single较Base+6.99，较对应single-1.80、AR_single较Base+7.95，较对应single-0.83。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

## Global-MMLU：完整模型×测试语言矩阵

本表每行使用该测试语言的全部固定正式样本，范围见上节。Base是原模型；SW/AR/RU single各指只经过相应语言一次完整训练的模型，对角线是对应语言训练。比较同一行；各语言题目/分词不能视为只改语言的严格因果对照。

| 测试语言 | Base | SW single | AR single | RU single |
|---|---|---|---|---|
| SW | 33.38 | 33.03 | 33.17 | 33.89 |
| AR | 47.69 | 52.30 | 52.29 | 54.63 |
| RU | 57.64 | 58.92 | 55.94 | 60.29 |

| 测试语言 | 训练模型 | 相对Base | 相对对应single |
|---|---|---|---|
| SW | SW_single | -0.35 | +0.00 |
| AR | SW_single | +4.61 | +0.01 |
| RU | SW_single | +1.28 | -1.37 |
| SW | AR_single | -0.21 | +0.14 |
| AR | AR_single | +4.60 | +0.00 |
| RU | AR_single | -1.70 | -4.35 |
| SW | RU_single | +0.51 | +0.86 |
| AR | RU_single | +6.94 | +2.34 |
| RU | RU_single | +2.65 | +0.00 |

SW：对应single较Base变化-0.35；其他语言训练后，AR_single较Base-0.21，较对应single+0.14、RU_single较Base+0.51，较对应single+0.86。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

AR：对应single较Base变化+4.60；其他语言训练后，SW_single较Base+4.61，较对应single+0.01、RU_single较Base+6.94，较对应single+2.34。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

RU：对应single较Base变化+2.65；其他语言训练后，SW_single较Base+1.28，较对应single-1.37、AR_single较Base-1.70，较对应single-4.35。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

## SIB-200：完整模型×测试语言矩阵

本表每行使用该测试语言的全部固定正式样本，范围见上节。Base是原模型；SW/AR/RU single各指只经过相应语言一次完整训练的模型，对角线是对应语言训练。比较同一行；各语言题目/分词不能视为只改语言的严格因果对照。

| 测试语言 | Base | SW single | AR single | RU single |
|---|---|---|---|---|
| SW | 40.20 | 25.98 | 25.49 | 25.00 |
| AR | 66.18 | 27.45 | 25.00 | 25.00 |
| RU | 71.57 | 33.33 | 25.98 | 25.98 |

| 测试语言 | 训练模型 | 相对Base | 相对对应single |
|---|---|---|---|
| SW | SW_single | -14.22 | +0.00 |
| AR | SW_single | -38.73 | +2.45 |
| RU | SW_single | -38.24 | +7.35 |
| SW | AR_single | -14.71 | -0.49 |
| AR | AR_single | -41.18 | +0.00 |
| RU | AR_single | -45.59 | +0.00 |
| SW | RU_single | -15.20 | -0.98 |
| AR | RU_single | -41.18 | +0.00 |
| RU | RU_single | -45.59 | +0.00 |

SW：对应single较Base变化-14.22；其他语言训练后，AR_single较Base-14.71，较对应single-0.49、RU_single较Base-15.20，较对应single-0.98。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

AR：对应single较Base变化-41.18；其他语言训练后，SW_single较Base-38.73，较对应single+2.45、RU_single较Base-41.18，较对应single+0.00。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

RU：对应single较Base变化-45.59；其他语言训练后，SW_single较Base-38.24，较对应single+7.35、AR_single较Base-45.59，较对应single+0.00。差值的不确定性见paired_changes.csv，不能仅看点估计断言相等或优劣。

## 非对应训练与对应single究竟差多少

下表比较同一测试语言上的完整固定样本；正差表示非对应训练较高，负差表示对应语言single较高。区间跨0不等于已证明两者等效；等效检验需要预定容许差距，本次不事后指定。SIB受归一化/类别偏好影响较强，其完整区间在CSV中保留。

| 测试语言 | 非对应模型 | 任务 | 非对应−对应 | 95%区间 |
|---|---|---|---|---|
| SW | AR_single | Belebele | +4.67 | [+1.77, +7.79] |
| SW | AR_single | XNLI | -8.78 | [-9.77, -7.78] |
| SW | AR_single | Global-MMLU | +0.14 | [-0.64, +0.90] |
| SW | AR_single | XL-Sum | -11.35 | [-12.20, -10.46] |
| SW | RU_single | Belebele | +3.67 | [+0.90, +6.51] |
| SW | RU_single | XNLI | -7.52 | [-8.47, -6.52] |
| SW | RU_single | Global-MMLU | +0.86 | [+0.13, +1.60] |
| SW | RU_single | XL-Sum | -7.09 | [-7.94, -6.25] |
| AR | SW_single | Belebele | -2.78 | [-4.78, -0.67] |
| AR | SW_single | XNLI | -3.07 | [-3.73, -2.47] |
| AR | SW_single | Global-MMLU | +0.01 | [-0.71, +0.68] |
| AR | SW_single | XL-Sum | -1.84 | [-2.14, -1.54] |
| AR | RU_single | Belebele | -1.33 | [-3.07, +0.33] |
| AR | RU_single | XNLI | +17.70 | [+16.61, +18.80] |
| AR | RU_single | Global-MMLU | +2.34 | [+1.69, +2.94] |
| AR | RU_single | XL-Sum | -1.41 | [-1.69, -1.15] |
| RU | SW_single | Belebele | +0.44 | [-1.25, +2.16] |
| RU | SW_single | XNLI | -17.47 | [-18.60, -16.35] |
| RU | SW_single | Global-MMLU | -1.37 | [-1.92, -0.79] |
| RU | SW_single | XL-Sum | -1.80 | [-2.02, -1.58] |
| RU | AR_single | Belebele | +0.33 | [-1.35, +2.13] |
| RU | AR_single | XNLI | -23.25 | [-24.51, -22.02] |
| RU | AR_single | Global-MMLU | -4.35 | [-5.01, -3.66] |
| RU | AR_single | XL-Sum | -0.83 | [-1.04, -0.62] |

## 选择题读出诊断与可回答的范围

choice_behavior.csv保留每个模型/语言/选择任务的金标与预测位置分布、最大预测类占比、未归一化ACC。SIB主分继续固定，raw仅作敏感性诊断；不能选择较高的一种评分。此前SIB已出现science/technology标签集中，选择题ACC提升也可能伴随答案位置偏好变化。这里不额外旋转正式题选项或据曝光测试挑提示。

若非对应训练也显著提升同一任务，能反驳“观察到的全部涨点只来自对应语言训练”的强说法；若对应训练仍有优势，说明当前数据/模型组合存在语言匹配相关收益。两者都不能单独识别指令学习：共享任务格式、选项读出、训练内容、底座已有语言能力以及双语监督均可能贡献。若要区分这些原因，需要独立开发材料和事先设计的内容/格式控制，本次不擅自新增训练。

本轮只做固定模型跨语言测试；一个训练seed，没有训练重复。测试已在旧实验暴露，本轮不调参数、提示或checkpoint；正式区间不校正多重比较，结论限于这些训练集和冻结协议。非选择题摘要用于观察是否只有选择题受益，但它本身也在训练任务混合中，不能当作未见任务泛化的证明。

## 归档与文件

已审计60/60项主分、90/90组配对区间；另有XQuAD16项分数与24组区间、英语8项分数与12组区间。新测评与复用结果均按同一固定模型身份核验，没有新增训练。

本目录公开关键聚合结果、区间与输出诊断；不重新发布测试题全文和服务器绝对路径。原始逐题生成、题号、输入身份哈希、冻结数据与源码继续保存在本地。模型身份、公开文件SHA256和源产物哈希见artifact_manifest.json；完整已有选择题/摘要设置见[前序固定协议](../sequential_lora_20261110/protocol.json)。

- [XQuAD生成式阅读结果与协议](xquad_report.md)
- [英语阅读与推理对照](english_anchor_scores.csv)，[配对区间](english_anchor_paired.csv)
- [全部五任务分数](scores.csv)，[全部模型配对区间](paired_changes.csv)
- [类别与答案位置诊断](choice_behavior.csv)，[SW摘要重复与长度诊断](summary_output_diagnostics.csv)

## 研究判断与后续定位（2026-11-12）

阅读任务存在强跨语言收益：RU Belebele上SW single82.44%、RU single82.00%；RU XQuAD F1分别64.77和65.83。当前数据支持共享任务/格式适配可能占有较大贡献，不支持把目标语言涨点全部归因于对应语言学习；一个seed、双语训练混合和不等词元预算仍限制机制解释。摘要对应语言均更好，XNLI迁移有方向性，因此不概括为所有任务与语言无关。

这组负结果削弱了将旧SW→AR→RU设置直接作为语言特异学习与遗忘主基准的价值。独立single交叉矩阵比较不同训练模型，顺序矩阵比较累积训练阶段，两者不能混为同一个对角线概念。持续学习定义本身不要求每行最高值在对应阶段，对角线非最大可来自前向迁移或本阶段负增益。后续优先核验任务定义和阶段增益，而不是以已暴露测试调出预期曲线。

另行运行的Mistral-7B-v0.1英文Alpaca→11语言MultiAlpaca全参数复现尚无P1/P2完整分数，不属于本目录已完成结果；仅有训练进度和Base分数不能证明两阶段方案有效，也不能单独因果区分指令与语言学习。
