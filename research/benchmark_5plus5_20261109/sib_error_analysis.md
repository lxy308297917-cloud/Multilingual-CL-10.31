# SIB标签偏好与错误分析

主指标固定为字符长度归一化ACC，原始ACC只作敏感性诊断，不选较高分。Macro-F1是七类等权平均；下表描述观测分布，不直接推断语义能力。

| 模型 | 归一化ACC | 原始ACC | Macro-F1 | 多数类基线 | 预测最多类别 | 该类预测数/204 |
|---|---:|---:|---:|---:|---|---:|
| sw_Base | 40.20 | 42.16 | 39.96 | 25.00 | geography | 89 |
| sw_historical_lora | 25.49 | 57.84 | 8.62 | 25.00 | science/technology | 201 |
| ar_Base | 66.18 | 67.65 | 65.88 | 25.00 | science/technology | 59 |
| ar_historical_lora | 25.00 | 37.75 | 5.71 | 25.00 | science/technology | 204 |
| ru_Base | 71.57 | 72.55 | 70.83 | 25.00 | science/technology | 67 |
| ru_historical_lora | 26.96 | 72.06 | 9.23 | 25.00 | science/technology | 200 |
| sw_all | 25.98 | 57.35 | 8.78 | 25.00 | science/technology | 202 |
| sw_full | 49.02 | 55.88 | 44.07 | 25.00 | science/technology | 111 |
| sw_kd | 28.92 | 55.88 | 15.68 | 25.00 | science/technology | 190 |
| sw_random | 50.00 | 55.88 | 44.82 | 25.00 | science/technology | 112 |
| sw_function | 51.96 | 55.39 | 47.05 | 25.00 | science/technology | 102 |
| ar_all | 25.00 | 50.98 | 5.71 | 25.00 | science/technology | 204 |
| ar_function | 25.00 | 34.80 | 5.71 | 25.00 | science/technology | 204 |

完整逐类召回和混淆矩阵见sib_diagnostics.json；numeric目录保留每题官方ID、金标签、两种预测及七个实际候选分数，可按固定官方源重建具体错误案例。204题的小样本差异需结合paired_changes.csv区间；区间只针对当前固定模型，不代表训练种子稳定性。SIB与Belebele重叠规模见sib_overlap_audit.json，不能将其作为独立文本证据叠加强化结论。
