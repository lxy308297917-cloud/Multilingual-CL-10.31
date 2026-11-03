# Hausa学习瓶颈诊断：完成交接

更新：2026-11-03T15:05:20.032604+08:00。旧基线快照 `ce6640a` 保持；本轮新增一个严格匹配HA阶段的Base→Hausa-only400步正对照，复用所有旧checkpoint。P0、全部正式测评、分析已完成；P1条件未建立，未训练新方法。

**结论：有少量Hausa知识收益，没有明确阅读收益；先学SW没有造成更差的最终HA成绩。应先检查目标任务学习/读出匹配和可复用跨语言收益为何回落，暂不把SW参数冲突作为新方法动机。**

| 任务 | Base A0 | HA-only A1 | SW FFT B0 | SW→HA B1 | A变化 | B变化 |
|---|---:|---:|---:|---:|---:|---:|
| HA阅读ACC /900 | 23.89 | 24.44 | 28.44 | 26.89 | +0.56pp | -1.56pp |
| HA知识ACC /14042 | 26.25 | 27.52 | 29.72 | 27.65 | +1.27pp | -2.07pp |

Hausa-only知识+1.27pp的95%区间[+0.55,+1.99]，阅读+0.56pp的区间[−2.33,+3.37]。顺序FFT最终阅读高2.44pp，知识相近但不证明等价。NLL改善不代替下游成效；全部指标及不确定性见各报告。

## 推荐阅读顺序

1. [研究决策：五问与一个待验证假说](next_method_decision.md)
2. [Hausa-only：实际配置、四模型对照、英语/SW变化](ha_only_control.md)
3. [全部学习曲线、logP、margin和训练loss](ha_learning_curves.md)
4. [评分、位置偏好、数据/任务与SST边界诊断](diagnosis.md)
5. [复现入口及公开资产边界](reproduce.md)

[完整正式表](../../results/next_stage/paper_table.md) · [四模型配对与起点分组](../../results/next_stage/ha-only-four-model-analysis-v1/analysis.json) · [全部开发汇总](../../results/next_stage/curves/analysis.json)

单seed42。开发阅读已曝光，知识0-shot开发与5-shot正式分开；全900/180/720均报告，不把720称历史未见。公开配置、来源/数据/模型哈希、scalar步日志、汇总、区间、PNG/PDF；完整日志、逐题与原生模型/optimizer留本地。旧结果不改，新入口优先于历史报告中的NOT_RUN。
