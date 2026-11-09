# 阶段性交接：三语言功能加权与FFT补充（实验进行中）

本快照记录2026-11-09已冻结的实验与已复核结果，**不是最终完成报告**。后续运行继续，不以本轮正式测试调系数、选checkpoint或停训。来源完整结果基于提交 `a183fea`。

## 1. 当前实际进度

- 三语言主线新增14组完整训练，已有2组完成并通过独立产物审计：SW Logit KD、RU均匀72点表示保持。另复用SW batch2普通LoRA。
- FFT补充新增2组完整训练：匹配普通FFT、FFT＋原36点表示保持。两组均通过3步GPU工程验证；完整400步训练仍在排队。
- 合计16组新增完整训练，当前2组完成。功能感知候选的正式分数尚未产生。
- SW Logit KD首批4项正式分数已完成逐题复核及配对区间；其余任务等待相同冻结协议测评。
- 历史FFT补测HumanEval+完成：60.98%（100/164），Base为78.05%（128/164）。这不是新增FFT保护组结果。
- 后续漂移／校准诊断已准备，**NOT_RUN**：只在全部113个正式测评阶段完成后申请GPU。

[主线进度](progress.md) · [三语言主表](results.md) · [strict／指令级补充](auxiliary_results.md) · [FFT设置与结果](../fft_representation_20261109/README.md)

## 2. 主线算法与比较条件

模型为官方 **Qwen/Qwen3-4B**，revision `1cfa9a7208912126459214e8b04321603b3df60c`；这里的Base是适配前官方模型参考，并非另一个名为Qwen3-4B-Base的模型。

SW、AR、RU是三次独立单语适配。每组6400条、固定Final400、seed42、assistant-only/non-thinking、cutoff2048、无packing。LoRA为全36层七类Attention/MLP投影、r16/alpha32/dropout0.05、micro2×GA8、有效batch16、LR1e-4、cosine/warmup0.05。

五方法：普通LoRA、均匀表示保持、同池Logit KD、随机加权表示保持、功能感知加权候选V1。

### 同一72点保护定义

每层记录post-attention residual与post-MLP residual，共72点；六组为前／中／后12层×Attention／MLP，每组12点。相同EN build128、参考回答最多4个预测位置、Base教师、固定capability归一化尺度、总系数1。每步8条旧样本，累计3200次曝光；check128不训练。

均匀、随机、功能组共同使用72点；均匀权重全部为1。旧SW36点FullRep另列，保留其强结果。**超过新72点对照，不自动等于超过旧36点强基线。**

功能权重只用固定build材料校准：将单组LoRA增量乘0.75，并以精确同总有效BA更新F范数的统一缩放作对照。六组risk为旧数学／指令相对CE恢复均值，减去目标阅读／摘要相对CE代价的正部分；分母为max(原NLL,0.1)。risk取正后分配，再投影到权重和6、各权重[0.5,2]；全部非正时用全1。规则固定，无性能准入门槛。

SW六权重（前Attention、前MLP、中Attention、中MLP、后Attention、后MLP）：

`[0.5824964788, 0.5824964788, 2.0, 1.3170968033, 0.9354137602, 0.5824964788]`

随机组使用相同权重多重集的预定排列，NumPy default_rng(61415)，不是另加训练种子。AR/RU按同一规则构造权重，当前待校准；其历史micro1 LoRA只作校准参照，主普通LoRA会按micro2重新对齐。

[完整算法与预算](current_objective.md) · [实现说明](implementation_clarifications.json) · [冻结记录](freeze.json)

## 3. 首批正式结果

均为百分比；行均为固定Final400，Base除外。历史36点组不是新72点对照。

| 模型 | SW Belebele | SW XNLI | EN Belebele | EN XNLI |
|---|---:|---:|---:|---:|
| Base | 27.33 | 33.19 | 62.89 | 49.24 |
| 普通LoRA | 36.33 | 42.22 | 87.56 | 71.06 |
| Logit KD | 40.22 | 41.06 | 87.44 | 63.07 |
| 历史36点FullRep | 40.89 | 44.25 | 88.44 | 70.34 |

Logit KD相对普通LoRA：

- SW Belebele：+3.89个百分点，按共享passage配对95%区间[+1.56,+6.26]。
- SW XNLI：−1.16个百分点，按共享premise区间[−2.32,−0.10]。
- EN Belebele：−0.11个百分点，区间[−1.53,+1.31]。
- EN XNLI：−7.98个百分点，区间[−9.14,−6.85]；它仍高于Base，不能把相对普通LoRA的差异称为对Base的遗忘。

这些是未作多重校正、固定seed42模型的样本配对区间，不是跨训练种子结论。XNLI选项偏好限制继续保留。摘要、知识、数学、IFEval、代码等未出齐，不能判断KD整体取舍，更不能宣称候选方法成功。

[逐项汇总](scores_long.csv) · [配对区间](paired_changes.csv) · [原始评分审计](score_audit.json)

## 4. FFT补充的作用与实现

目标是检验原36点Full Representation能否迁移到全参数训练。它**不是功能加权FFT**。

已实际核验历史FFT：原始SW6400（不是清洗GV／NLLB替换版）、400步、micro1×GA16、LR1e-5、BF16、AdamW／DeepSpeed CPUAdam offload、cosine/warmup0.05。历史完整权重保留，并另补同实现普通FFT，避免工程变更与保护效果混淆。

新增两组均采用非重入梯度检查点、一次联合loss/backward；保护组沿用原P2缓存36点、数学/指令尺度及lambda1。每步8条旧样本，与LoRA对齐，不因GA16翻倍。工程验证确认全部398个唯一参数张量收到有限梯度、CPUAdam真实LR绑定、48次target backward对应3次更新，24GB卡可运行。正式400步重新从Base初始化。

Qwen3共享embedding/readout；范数审计验证保存的重复head与embedding相同，只计一次。保护特征位于最终norm之前，不能假设它直接约束所有读出变化。

[FFT主表](../fft_representation_20261109/results.md) · [strict／指令级／BLEU](../fft_representation_20261109/auxiliary_results.md) · [工程验证](../fft_representation_20261109/engineering_audit.json) · [比较限制](../fft_representation_20261109/limitations.md)

## 5. 评测与交付边界

目标4项：Belebele、XL-Sum、Global-MMLU、XNLI。通用EN6项：GSM8K、官方IFEval、MMLU、HumanEval+、Belebele、XNLI。Global-MMLU各语言全14042题／57学科／5-shot；阅读与XNLI使用冻结原任务形式。

FFT完整表同时保留双向FLORES chrF++／BLEU、GSM8K flexible/strict、IFEval prompt/instruction strict。指令级IFEval以834条指令为分母，配对区间重采样541个完整prompt；翻译按语料统计量重算，不平均句级BLEU或chrF。

本阶段GitHub快照提供代码、固定配置、已验证汇总和区间，以及已完成单元格的逐题数值矩阵。**覆盖所有模型的完整矩阵、最终报告和全部诊断将在正式实验完成后补齐**。本机原始逐题预测、失败尝试与checkpoint保留，未完成结果留空而非0。

诊断将对Base及21个已有适配权重运行旧英语check128推理，输出36层／72点漂移和assistant-token NLL。check材料曾用于旧P1诊断，不能声称历史完全未暴露；辅助指标不代替下游成效，也不反馈选模。

当前是独立单语适配，**不是顺序多语言CL的有效性证明**。单seed、单个随机排列、语言间数据差异、calibration CE的代理局限、旧36点与新72点差异、额外教师信息和成本均须保留。FFT目前未包括同池FFT CE／KD对照；LoRA对应结果不能替代它们。

[限制](limitations.md) · [实际成本](compute_costs.json) · [实测环境](environments.json)

## 6. 供讨论的核心问题

1. 在完整矩阵上，功能权重能否超过均匀72点、随机同多重集、同池KD，并保留旧36点FullRep这个强参照？
2. 改善是否伴随目标摘要或知识收益损失、更新范数缩小，或只改善某种输出形式？
3. 原36点表示保持在FFT上的结果，支持哪些适用范围，又留下哪些未控制的参数化／读出／额外数据因素？

这些问题由本轮固定预算正式结果回答；目前不追加方法、不改lambda、不增加训练种子，也不挑正式test最优点。

## 快照补充

AR Base全量Global-MMLU已完成并独立复核：47.6855%（6696/14042，57学科，5-shot）。本阶段已有51个主线主表单元格（含历史参照）和10个补充单元格；FFT比较表有62个主要单元格和18个补充单元格。它们含复用模型，不代表已完成全部新训练。

公开数值可在不加载模型的情况下复算现有均值／语料分数及配对区间：

```bash
python scripts/function_aware_formal_20261109/verify_numeric.py research/function_aware_formal_20261109 --family fa
python scripts/function_aware_formal_20261109/verify_numeric.py research/fft_representation_20261109 --family fft
```

`all_expected_cells_present=false`明确表示实验尚未完成；数值一致性PASS只覆盖当前导出数据。原始题目、模型和执行身份由独立服务器审计记录支持。
