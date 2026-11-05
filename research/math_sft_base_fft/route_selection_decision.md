# 冻结门槛的路线决定

状态：FAIL_FIXED_E1_AND_E2_GATE。同一epoch通过语言数：{'1': 1, '2': 1}。

每语言MGSM与MSVAMP各至少+2pp，至少一项配对95%下界>0；同一epoch>=2/3语言即A通过。全部九语言Base、三语言两节点和所有通用结果保留。英语下降不构成失败。路线B：AUTHORIZED_CONDITIONAL_READY_FOR_RESOURCE_AND_DATA_AUDIT。

|路线|训练／评测状态|冻结门槛与决定|
|---|---|---|
|A：全部有效MathOctopus×2轮|COMPLETE|FAIL_FIXED_E1_AND_E2_GATE；同一epoch通过数：{'1': 1, '2': 1}|
|B：mCoT-MATH 30k×2轮|NOT_RUN|AUTHORIZED_CONDITIONAL_READY_FOR_RESOURCE_AND_DATA_AUDIT|

这是已暴露benchmark引导的探索性决策，未作独立盲测或多重比较校正。
