# Benchmark v2 精选结果

16组MT/SUM正式分数与8组Code生成/静态结果。Code pass@1未运行，null不是0。

- paper_table.md / pilot_table.md：正式与工程子集分开。
- results.json / results.csv：逐任务指标、模型身份、协议、成本及配对区间。
- individual/：原评分summary原字节副本。
- run_manifests/：科学字段投影；原运行manifest SHA保留，主机路径已移除。
- sample_ids/：按manifest顺序合并各分片即实际完整测试ID。
- verification.json / identity_audit.json：四模型实际提示、数据和参数核验，以及模型实体和历史文件重新哈希。
- data_audit.json / legacy_comparison.json：明确重合与精确匹配范围，不证明广泛无污染。
- output_diagnostics.json：输出行为与长度关联，非因果机制分析。
- resource_report.json：包括失败尝试的真实墙钟与采样显存，不是CUDA kernel时间。

原始逐条预测、完整新闻文章与权重留在研究服务器，不在此分发。重新生成需获取准确Final权重并遵守原数据许可。
