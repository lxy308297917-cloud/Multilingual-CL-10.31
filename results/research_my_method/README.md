# 公开派生结果索引

## 正式任务

- [全部固定任务分数及实际状态](sequential/summary.json)与[可读任务表](sequential/tables.md)。历史500例与full1012翻译分开；不混合平均ACC/ROUGE/chrF。
- [SST完整报告](../../research/sst_reproduction/reproduction_report.md)、NoRefresh[对Base配对区间](../sst/no_refresh/paired_vs_base.json)、[对FFT配对区间](../sst/no_refresh/paired_vs_fft.json)、[翻译/摘要配对区间](../sst/no_refresh/sum_mt_paired.json)。
- R10[对Base配对区间](../sst/r10/paired_vs_base.json)、[对NoRefresh配对区间](../sst/r10/paired_vs_no_refresh.json)、[翻译/摘要配对区间](../sst/r10/sum_mt_paired.json)。
- [FFT两阶段报告](../../research/my_method_20261103/sequential_results.md)与[全部21项阶段配对](sequential/fft_paired.json)。
- [SST-NoRefresh全部21项阶段配对](sequential/sst_paired.json)，与自身W1比较；[实际曝光一致性](sequential/sst_exposure_alignment.json)。

## 有限机制探索

- [配对系数pilot报告](pilot/report.md)与[分析](pilot/analysis.json)。逐样本预测仅保留本地，公开[行数与SHA256](pilot/local_predictions_identity.json)。
- [tied embedding有限控制报告](tied_embedding/report.md)与[分析](tied_embedding/analysis.json)。
- [FFT阶段几何](geometry/fft_sequential.md)、[实际NoRefresh Final精度与空间审计](../sst/no_refresh/master_export_geometry.json)、[R10 step100中途投影器](../sst/r10/interim_step100_projector_geometry.md)。几何不是功能成效。
- [fit内事后稳定性](pilot/fit_stability_posthoc.md)，不改变主要功能阴性结果。
- [六模型完整正式权衡图](../sst/figures/baseline_tradeoffs.png)。机制图在geometry/figures、pilot/figures；../sst/no_refresh/figures为较早的NoRefresh完整快照。旧工程step20不冒充Final。

## 审计

[curation_manifest.json](curation_manifest.json)记录原件与公开投影SHA256；私人路径已置换。SST恢复、实际Adam元数据、曝光对齐、耗时/范数/显存均保留独立JSON。CPU/真实分布式验收和限制见[实现说明](../../research/sst_reproduction/implementation_notes.md)。
