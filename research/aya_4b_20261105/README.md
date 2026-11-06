# Qwen3-4B SW适配：本轮交付完成

四模型主线80项分数、60项配对比较，补充XNLI/XCOPA12项分数、9项配对比较均已完成。两个预定翻译修正候选也已完成，但均未达到冻结开发条件，尚未证明修复成功。

- [完整正式报告](final_report.md)
- [交给ChatGPT的总结](chatgpt_handoff.md)
- [最终研究结论及讨论问题](final_decision.md)
- [主结果和配对区间](results.md)
- [92项合并分数](../../results/aya_4b_20261105/all_scores.csv)
- [69项合并配对比较](../../results/aya_4b_20261105/all_paired_comparisons.csv)
- [复现说明](reproduction.md)
- [补充XNLI/XCOPA](supplemental_benchmarks.md)
- [XNLI提示敏感性诊断](xnli_format_diagnosis.md)
- [第一个翻译候选](translation_rebalanced_candidate_result.md)
- [第二个翻译候选](translation_lr2e5_candidate_result.md)
- [历史阶段记录（非当前状态）](progress.md)

只有seed42；样本区间不代表跨训练种子稳定性。模型规模与数据同时改变，不能分别归因。原有负面结果完整保留；候选正式确认NOT_RUN，第三候选、其他语言或MyMethod未开展，旧路线B仍停止。
