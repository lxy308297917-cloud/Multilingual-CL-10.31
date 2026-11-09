# ChatGPT阶段交接：5+5与XQuAD补测（仍在运行）

## 历史IFEval主指标更正（2026-11-10）

此前新5+5汇总错误地用历史IFEval的instruction-level strict覆盖了prompt-level strict，原始测评和旧历史报告没有改变。主表现在固定采用541个提示的prompt_level_strict_acc：SW 49.35%、AR 42.70%、RU 42.33%；62.47%、56.47%、56.35%是834条指令的补充指标，不能混用。历史三行的Base差与GeneralAvg5已重算，其他任务的定义不变。公开更正前数值与原因见 [更正记录](metric_correction.json)。解析现按任务＋指标双键取值，并拒绝重复主指标；独立复算器核对18个历史主指标单元格。

2026-11-10T01:30:42.114011+08:00

本轮优先补齐已存在checkpoint的下游正式分数；没有新增训练或训练种子。五个待启动训练与机制诊断、额外翻译已按用户要求暂缓，不自动恢复。原研究目标中这些比较仍未完成，不把本轮补测阶段当作全部My Method研究完成。

## 最新完成项与尚未完成范围

2026-11-10T03:29:13.571761+08:00

- [三语言历史基线完整5+5](baseline_report.md)：6个模型语言行、60个主表单元格已齐；三语阅读/摘要提高，数学/指令/代码下降，英语阅读与MMLU提高。SIB负结果和归一化敏感性完整保留。三语独立单语适配，不是顺序CL或多种子证据。
- [SW候选完整端点与强对照](../function_aware_formal_20261109/chatgpt_handoff.md)：候选10项已齐；相对普通LoRA有收益和通用恢复，但尚无超越随机/原36点强参考的明确总体证据，均匀72点及KL等对照仍有缺项。
- [FFT当前排定补测](../fft_representation_20261109/README.md)：10阶段正常结束；76项主分数和20项补充分数已复核，所有剩余缺项均属用户暂缓。新保护组仍有IFEval/代码遗忘，匹配普通FFT与额外翻译不恢复。
- SIB22端点及[XQuAD10端点](../xquad_20261109/final_report.md)全部完成。XQuAD AR候选未明确超过普通LoRA，RU候选F1低2.12pp，负结果保留。
- 主表冻结快照166/220个可执行单元格；其余主线对照与AR/RU方法网格继续运行。主训练原16项为11完成、5用户暂缓，无新训练；整个研究目标尚未全部完成。

下方“首批”表保留为早期阶段记录，最新完整基线以baseline_report.md和results.md为准。

## 固定评测体系

目标5：Belebele ACC、XNLI ACC、XL-Sum ROUGE-Lsum、Global-MMLU ACC、SIB-200 acc_norm。通用5：GSM8K flexible、IFEval prompt strict、HumanEval+、EN MMLU、EN Belebele。TargetACC4只平均四项ACC，XL-Sum另列；缺项不算平均。GeneralAvg5是描述性的异质能力等权均值，不能代替逐项结果。EN XNLI、strict子指标与既有翻译分数保留在旧轨道。

SIB主指标固定为标准Harness字符长度归一化似然ACC；未归一化ACC是辅助列，不根据正式结果选高分。XQuAD只作为AR/RU补充，不加入5+5平均；SW官方不支持。XQuAD采用原生Harness语言提示、non-thinking、greedy128、换行停止、官方SQuAD v1.1 EM/F1。

## 首批真实结果：AR知识与问答

历史AR LoRA（micro1、Final400）全量Global-MMLU为 **47.69→52.14%**，14042题、57学科、5-shot；相对Base +4.46个百分点，同题配对95%区间[+3.61,+5.24]。这是已有正确结果的补齐，没有重新训练。

| XQuAD AR | EM% | F1% |
|---|---:|---:|
| ar_Base | 0.08 | 26.46 |
| ar_historical_lora | 49.33 | 66.71 |

上述问答增益**包含输出格式学习，不能全部归因于语义理解增强**。1190题全量诊断：Base有611条带答案标签，LoRA为0；预测/参考字符长度中位比由5.025变为1.0。按相同规范化规则，完整参考词串出现在预测中的数量607→710；这个包含统计不是准确率，可能计入复述，不能替代官方EM/F1。主提示、生成与评分没有修改，格式诊断为首次Base结果暴露后的辅助分析。问答区间以240段文章分簇、2000次重采样。

## SIB-200出现明显评分敏感性

| 历史LoRA | 归一化ACC Base→LoRA | 原始ACC Base→LoRA |
|---|---:|---:|
| SW | 40.20→25.49 | 42.16→57.84 |
| AR | 66.18→25.00 | 67.65→37.75 |
| RU | 71.57→26.96 | 72.55→72.06 |

SW/AR历史LoRA在归一化评分下分别有201/204、204/204题预测为第一类science/technology。该类也是最长候选，不能仅凭这些结果分离标签顺序、长度归一化和模型更新各自的因果作用。逐题似然重算与框架一致，不是自由生成答案提取器误判。保留所有负结果，不删除SIB、不切换到较高的原始ACC当主指标。

### SW当前方法的SIB初步对照

| 模型 | 固定主指标ACC | 原始ACC |
|---|---:|---:|
| sw_all | 25.98 | 57.35 |
| sw_full | 49.02 | 55.88 |
| sw_kd | 28.92 | 55.88 |
| sw_random | 50.00 | 55.88 |
| sw_function | 51.96 | 55.39 |

功能候选主指标51.96，高于均匀72点49.02和随机50.00，但原始ACC并未超过普通LoRA；204题的同题配对区间为：候选相对均匀72点+2.94pp，[+0.98,+5.39]；相对随机+1.96pp，[-0.49,+4.41]（2000次重采样、未作多重比较校正）。尚不能证明方法总体优于强基线。旧36点FullRep对照仍将保留；新的阅读、摘要和通用结果未齐，不能仅凭SIB决定方法成败。

## 依赖与解释边界

- SIB与Belebele源文本存在大量包含关系：SW检测178/204、AR170/204、RU171/204；不是独立文本证据。训练6400例未检出完整SIB文本包含，不能排除片段、改写或预训练曝光。
- 历史micro1与新P2/主线micro2严格分行；三语言是独立单语适配，不是顺序CL或三个训练种子。
- 本轮仍只seed42；所有置信区间反映固定模型/当前样本，不是跨训练种子稳健性，没有多重比较校正。
- 新匹配FFT、AR均匀/KD/随机、RU KD被用户暂缓。保持这些未完成条件可见，不能把原16组训练表说成全部完成。
- 所有允许继续收尾的训练已结束；当前只安排指定benchmark，不占空卡。账号总量临时最多5卡，其他用户任务不动。

## 结果与复现入口

- [5+5完整持续表](results.md)、[逐项Base差值](base_deltas.csv)、[跨语言ACC4宏平均](cross_language_macro.csv)。
- [SIB错误与标签偏好](sib_error_analysis.md)、[混淆矩阵](sib_diagnostics.json)、[原始评分复核](score_audit.json)、[配对统计](paired_changes.csv)。
- [Global-MMLU各模型57学科](gmmlu_subjects.csv)、[成本](evaluation_costs.json)、[优先级与暂缓范围](priority_scope.json)。
- [XQuAD结果](../xquad_20261109/results.md)、[原始EM/F1审计](../xquad_20261109/score_audit.json)、[输出格式诊断](../xquad_20261109/format_diagnostics.json)。
- 配置及数据revision/hash见 `configs/benchmark_5plus5_20261109` 和 `configs/xquad_20261109`；只改账号资源上限的适配器保留原科学评测源码哈希，不改任务协议。

本文件是本地正在进行的阶段快照，尚未声称全部补测完成；GitHub最终提交将在完整结果与审计之后更新。
