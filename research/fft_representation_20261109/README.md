# FFT＋36层全表示保持补充实验

> 最新用户优先级：只优先5+5与XQuAD。FFT保护组已自然完成400步并通过训练/权重审计；匹配普通FFT在启动前暂缓。额外翻译与机制诊断暂缓，旧屏障服务已停，不自动恢复。下文原始设计仍保留，不代表所有计划已完成。

依据用户2026-11-09上传的新FFT任务，优先补齐SW上的跨参数化验证。既有三语言LoRA任务保留，不覆盖。

- 历史FFT完整400步、6400条和两个权重文件已重新核验，另列历史行。
- 新增匹配普通FFT与FFT＋全表示保持，各完整400步。两组均采用非重入梯度检查点及单次联合反向传播，避免ZeRO2双反向的更新边界与重复梯度归约问题。改变实现的普通FFT也补跑，避免与保护效果混淆。
- 保护组使用P2原始36层特征，不是新六组72点特征；同Base缓存、EN build128、固定尺度与lambda1，check128不训练。每步8条旧样本，曝光量与P2对齐。
- 3步工程验证采用已有最长16条目标样本，检查显存、全参数梯度、非零表示损失、CPUAdam实用LR和单次反向。只检查代码，不读取benchmark；正式400步重新从Base开始。
- 两组FFT完整测SW/EN阅读、SW摘要、SW知识、EN MMLU、GSM8K strict/flexible、IFEval prompt/instruction strict、XNLI SW/EN、HumanEval+及双向FLORES；历史完整分数仅在协议匹配时复用。翻译负结果保留。
- 最终同时展示Base、历史FFT、匹配FFT、新FFT＋表示保持、P2 LoRA/FullRep/CE Replay、历史SSU，标明学习率、microbatch、可训练参数及额外成本差异。

两组3步GPU工程验证均已通过；当前仅保护组完成完整400步，匹配普通FFT由用户暂缓，不恢复。所有输出独立保存，不因正式分数早停、改系数或选checkpoint。

## 持续更新与审计入口

- [完整八行对照结果](results.md)：缺失标—，历史与新训练分开。
- `progress.json`：实际队列及工程/正式训练阶段。
- `score_audit.json`、`paired_changes.csv`、`numeric/`：已完成单元格的原始评分复核和可复算配对统计；未完成部分明确保留。
- `full_update_audit.json`：CPU参数变化审计，区分共享embedding/readout、最终norm与decoder；不作方法成效依据。
- [比较边界](limitations.md)：保护特征、额外数据和跨参数化限制。

## 环境与执行入口

沿用[本轮三套实测环境](../function_aware_formal_20261109/environments.json)。`run.py --mode none` 是普通匹配FFT，`run.py --mode full36` 是36点表示保持FFT；均先工程验证，再从Base完整训练400步。训练环境需有与torch/CUDA ABI一致的CPUAdam扩展；本机复用了已有编译产物，没有改动共享Python环境。

`collect_results.py`、`paired_results.py --partial`、`audit_full_updates.py` 与 `compute_costs.py` 是CPU结果整理和审计入口。它们不决定选模，不恢复训练。

完整要求还包括 [GSM8K strict、IFEval instruction strict与双向BLEU](auxiliary_results.md)，它们与主表同时保留。IFEval指令级区间按完整prompt重采样，不把834条指令当作独立样本。`numeric/`及`auxiliary_numeric/`提供带输入ID哈希的逐题数值矩阵；模型列可直接作配对差，保留原始逐题预测用于本地错误分析。

尚待完成的机制部分：固定Final的36层旧英语漂移及数学／指令校准诊断。本轮尚未运行这部分推理；正式benchmark优先，诊断不会用于重新调系数或选择checkpoint。

## 正式测评之后的机制诊断

已准备并冻结 `diagnose_final.py` 与模型清单：仅在本轮全部113个正式测评阶段完成后，才允许申请GPU，对Base及已有21个适配权重作推理诊断。使用既有P1 check128（数学／指令各64条，曾用于旧诊断），与训练build128不重叠；不新增训练，不用于调系数或选checkpoint。

输出固定Final的36层／72点表示漂移及参考回答NLL，作为辅助解释，与下游成效表分开。当前状态为 **NOT_RUN，等待正式benchmark**。新增教师缓存与诊断耗时另计，不把NLL或低漂移当作方法收益。

离线复算入口为 `scripts/function_aware_formal_20261109/verify_numeric.py`，传入本目录并指定 `--family fa` 或 `--family fft`。当前部分单元格可复算通过；`--require-complete`会要求所有预定单元格齐全。验证范围是导出数值的一致性，原始输入与模型身份由独立服务器审计支持。

## 新增目标任务阶段结果（固定Final400）

2026-11-10T02:10:44.946906+08:00

| 模型 | SW阅读 ACC% | SW摘要 ROUGE-Lsum | SW知识 ACC% | SW XNLI ACC% |
|---|---:|---:|---:|---:|
| 原模型 | 27.33 | 13.71 | 33.38 | 33.19 |
| 历史普通FFT | 39.78 | 18.36 | 32.72 | 49.00 |
| 新FFT＋36点表示保持 | 42.56 | 19.04 | 32.97 | 44.87 |

新保护组相对Base的阅读、摘要、XNLI样本配对区间不跨零；知识为-0.41pp，区间[-1.34,+0.61]，没有明确知识收益。没有按这项负结果换指标、改系数或选checkpoint。通用能力仍未全部出齐，不能判断完整学习—保持取舍。

历史普通FFT与新实现使用不同梯度检查点/反向封装；严格匹配的普通FFT尚未运行（USER_DEFERRED）。历史差异只能描述，不能把它视为隔离表示保持效应的因果对照。当前是原36点保护移植到FFT，不是功能风险加权FFT，也不是已经证明新方法的结果。完整逐题核验见score_audit.json，配对区间见paired_changes.csv；区间仅针对当前固定模型，未作多重校正。
