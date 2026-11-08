# 三语言LoRA更新消融与功能表示诊断

18组训练后消融全部完成并复核。没有新增SFT；消融不是部分层训练或LinguaMap严格复现。

- [研究交接](chatgpt_handoff.md)
- [完整报告](final_report.md)
- [全部绝对分数](results.md) / [配对变化与区间](ablation_report.md)
- [参数更新分布](norm_report.md) / [评分限制](reading_protocol_caveats.md)
- [设置](experimental_design.md) / [协议](protocol_summary.json) / [成本](resource_costs.md)
- [公开代码和复现边界](../../scripts/lora_layer_analysis/README.md)
- [最新功能表示阶段结果](../functional_representation_20261109/README.md)

正式测试已暴露，单seed42；不用于挑选层数、系数或训练方法。所有正负结果保留。
