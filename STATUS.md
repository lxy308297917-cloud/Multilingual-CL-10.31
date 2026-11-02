# 当前状态：内部机制诊断已完成计算

## Benchmark v2 补充测评（2026-11-02完成）

固定Base/FFT/LoRA/SSU Final，全量生成60,352条；16组翻译/摘要正式评分及8组代码生成/静态检查完成，28项测试通过。SW摘要和EN→SW有收益，SW→EN下降，EN摘要没有一致遗忘。代码pass@1为NOT_RUN；CrossSum仅可行性审计。原机制诊断负面结果保留。

- [Benchmark v2报告](research/bench_v2/benchmark_report.md)
- [正式表与配对区间](results/bench_v2/summary/paper_table.md)
- [复现与限制](research/bench_v2/reproduce.md)

固定seed42 Base/FFT/LoRA/SSU正式80指标、141项完成快照与旧结果保留。三组参数/激活审计、13,224条模块干预预测、1,224行条件效应、72行交互、最终v3激活补丁均完成。37项CPU测试通过。未取得明确功能分离证据，未设计或训练最终My Method。

阅读[完整报告](research/mechanism/mechanism_report.md)及[早间摘要](research/mechanism/morning_brief.md)。已取消的候选四组与未完成多种子复测保持停止。本公开副本不包含服务器运行管理信息、原始数据或权重。
