# 当前完成状态：220/220及XQuAD10端点

当前可运行的22个模型/语言端点全部完成5+5，XQuAD的AR/RU共10端点也完成；GPU已释放，无接续任务。5个暂缓训练、额外翻译和机制诊断仍未恢复，原研究整体范围未全部完成。

- [完整正式报告](../function_aware_formal_20261109/final_report.md)
- [ChatGPT交接](chatgpt_handoff.md)
- [完整结果表](results.md)
- [阶段验收](priority_completion.json)

以下保留此前的设置与阶段记录；其中历史“运行中/未完成”不表示当前继续运行或恢复暂停任务。

# 五项目标语言＋五项通用能力补测

## 历史IFEval主指标更正（2026-11-10）

此前新5+5汇总错误地用历史IFEval的instruction-level strict覆盖了prompt-level strict，原始测评和旧历史报告没有改变。主表现在固定采用541个提示的prompt_level_strict_acc：SW 49.35%、AR 42.70%、RU 42.33%；62.47%、56.47%、56.35%是834条指令的补充指标，不能混用。历史三行的Base差与GeneralAvg5已重算，其他任务的定义不变。公开更正前数值与原因见 [更正记录](metric_correction.json)。解析现按任务＋指标双键取值，并拒绝重复主指标；独立复算器核对18个历史主指标单元格。

入口：[当前设置](current_objective.md)、[持续结果](results.md)、[SIB诊断](sib_error_analysis.md)、[优先级变更](priority_scope.json)。

目标5：Belebele、XNLI、XL-Sum、Global-MMLU、SIB-200；通用5：GSM8K、IFEval、HumanEval+、EN MMLU、EN Belebele。旧EN XNLI、strict子指标和翻译成绩保留在原轨道，未删除。

补测SIB三语言和历史AR/RU的全量Global-MMLU，复用已核实的同模型同协议成绩。SIB27个原计划模型语言单元中5个未训练模型被用户明确暂缓，因此当前实际可完成22个；不为补表重新训练。旧模型和新P2模型按身份分行。

TargetACC4只平均四项ACC；摘要独立报告。SIB按已冻结的标准Harness acc_norm字符长度归一化评分，原始ACC另报，禁止选较高分。SIB与Belebele存在大量文本包含重叠，详情见sib_overlap_audit.json；未发现当前6400例中包含完整SIB测试文本，但该检查不能排除改写、片段或预训练曝光。

本次临时允许账号总量5卡，仅测评利用新增槽位；正在运行的训练正常结束，暂停训练和诊断不自动恢复。所有任务完成后释放GPU，不占空卡。

已完成基线请看 [三语言历史5+5完整报告](baseline_report.md) 和baseline_completion.json；完整主线方法网格仍在进行。
