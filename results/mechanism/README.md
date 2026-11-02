# 机制诊断结果

三组主要计算全部完成。正式 benchmark 与本目录诊断分开；原 baseline 表保持原样。

- [完整报告与Q1–Q4结论](../../research/mechanism/mechanism_report.md)
- [早间汇报摘要](../../research/mechanism/morning_brief.md)
- [恢复与注入分面散点图](report_figures/restore_vs_inject.svg)，42个点，参考分别为Final和Base，不将Final−inject解释为模块贡献。
- parameter：逐层/组件有效权重变化CSV，描述性证据，不等于重要性。
- activation：60个相同输入的Base/Final差异，各2352行，prompt/teacher-forced参考区域分开，不是跨语言token对齐。
- activation_patching：最终v3各368行、26样本，包括26例中性语义控制及14例真实不同题等长控制；参考LL不是生成准确率，中性控制可能分布外。
- conditional_effects / interaction_contrasts：每模型408/24行；重复提示与pair ID聚类，探索性配对bootstrap区间，不做多重校正。
- math_response_audit：事后保守数值提取，不改原评分，未知不视作错，范围不是置信区间。
- reference_scores：同一232输入的Base/Final分任务统计。
- verification_summary：37项CPU测试及三组真实运行核验摘要，无私有运行管理信息。

三组没有任何字面模块恢复条件的经验95%区间排除0。小样本不证明无效，也不支持宣称发现语言/任务分离。FFT/SSU完整Block还有LayerNorm变化，不能把全部对比视为纯Attention-MLP交互。SSU在合成指令子集未优于FFT，应结合正式IFEval并保留协议差异。

公开副本提供代码、科学配置、来源/选题ID、汇总与逐条件统计；未发布权重、原始数据文本、完整生成日志。完整重跑需要外部取得模型与数据，并重建few-shot排除材料，不宣称外部机器已经复现。
