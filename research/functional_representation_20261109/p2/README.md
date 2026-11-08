# P2：训练中的表示保持与同池回放

SW三组新训练均完成固定Final400和全量冻结端点评测。Base及旧batch1 Full为历史同协议参考，新batch2 All-layer是主要训练对照。

- [给ChatGPT的交接](chatgpt_handoff.md)
- [完整报告](final_report.md) / [下一阶段方法决策](next_method_decision.md)
- [全部结果与48项配对比较](results.md)
- [训练配方及工程边界](training_protocol.md)
- [公开代码与复算入口](../../../scripts/functional_representation/p2/README.md)
- [目标数据逐样本哈希](target_exposure_reference.jsonl) / [旧池清单与哈希](old_pool_manifest.json)
- [实际测评命令的可移植记录](evaluation_commands.json) / [环境](environment.json)
- [额外资源与成本](cost_summary.json) / [数值独立复算](public_numeric_recomputation.json)

完整保留正负结果。单SW、单seed42、单阶段适配；测试已历史暴露，未用于本轮调系数或选择checkpoint。尚非顺序CL或创新方法有效性的证明。
