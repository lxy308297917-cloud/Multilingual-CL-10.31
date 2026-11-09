# 五项目标语言＋五项通用能力补测

入口：[当前设置](current_objective.md)、[持续结果](results.md)、[SIB诊断](sib_error_analysis.md)、[优先级变更](priority_scope.json)。

目标5：Belebele、XNLI、XL-Sum、Global-MMLU、SIB-200；通用5：GSM8K、IFEval、HumanEval+、EN MMLU、EN Belebele。旧EN XNLI、strict子指标和翻译成绩保留在原轨道，未删除。

补测SIB三语言和历史AR/RU的全量Global-MMLU，复用已核实的同模型同协议成绩。SIB27个原计划模型语言单元中5个未训练模型被用户明确暂缓，因此当前实际可完成22个；不为补表重新训练。旧模型和新P2模型按身份分行。

TargetACC4只平均四项ACC；摘要独立报告。SIB按已冻结的标准Harness acc_norm字符长度归一化评分，原始ACC另报，禁止选较高分。SIB与Belebele存在大量文本包含重叠，详情见sib_overlap_audit.json；未发现当前6400例中包含完整SIB测试文本，但该检查不能排除改写、片段或预训练曝光。

本次临时允许账号总量5卡，仅测评利用新增槽位；正在运行的训练正常结束，暂停训练和诊断不自动恢复。所有任务完成后释放GPU，不占空卡。
