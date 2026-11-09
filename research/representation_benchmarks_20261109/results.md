# 四模型完整固定端点成绩

共同 seed42、batch2、Final400；Base未适配。摘要为ROUGE-Lsum，其余为百分比。主要表不含XNLI辅助轮换，不混合单位求平均。

|任务／指标|Base|普通LoRA|全表示保持|CE回放|
|---|---:|---:|---:|---:|
|Belebele target / ACC|27.33|36.33|40.89|37.00|
|Belebele EN / ACC|62.89|87.56|88.44|87.22|
|XL-Sum target / ROUGE-Lsum|13.71|20.18|20.58|20.84|
|GSM8K / flexible-extract|84.46|81.05|84.76|62.47|
|IFEval / prompt_level_strict_acc|78.37|48.43|79.85|59.52|
|Global-MMLU SW / ACC|33.38|33.03|35.34|33.75|
|XNLI SW / ACC|33.19|42.22|44.25|33.45|
|XNLI EN / ACC|49.24|71.06|70.34|35.27|
|HumanEval+ / pass@1|78.05|60.37|74.39|62.80|

## 辅助指标

|任务／指标|Base|普通LoRA|全表示保持|CE回放|
|---|---:|---:|---:|---:|
|GSM8K / strict-match|61.03|63.76|75.59|61.03|
|IFEval / inst_level_strict_acc|84.65|61.27|86.09|69.78|
|HumanEval 原始测试 pass@1|84.76|65.24|80.49|69.51|

## XNLI：固定标签轮换诊断

不是替代主要协议的更优分数。历史Base已有选项偏好；本轮相同规则用于全部模型。

|模型|SW主要|SW轮换|EN主要|EN轮换|
|---|---:|---:|---:|---:|
|Base|33.19|36.99|49.24|64.93|
|all_layer|42.22|43.53|71.06|71.86|
|full_feature|44.25|44.95|70.34|72.91|
|ce_replay|33.45|34.69|35.27|41.88|

## 新增任务配对区间

区间单位为百分点；XNLI以共享premise聚类重采样，SW1669群、EN1670群；代码以164道题重采样。均为固定模型的2000次配对bootstrap，不是训练种子置信区间。

|任务|比较（后者减前者）|差值|配对95%区间|旧对→新错|旧错→新对|
|---|---|---:|---|---:|---:|
|XNLI SW|Base → all_layer|+9.02|[+8.00, +9.99]|824|1276|
|XNLI SW|Base → full_feature|+11.06|[+10.01, +12.10]|547|1101|
|XNLI SW|Base → ce_replay|+0.26|[-0.20, +0.72]|142|155|
|XNLI SW|all_layer → full_feature|+2.04|[+1.02, +3.01]|353|455|
|XNLI SW|ce_replay → full_feature|+10.80|[+9.69, +11.90]|583|1124|
|XNLI EN|Base → all_layer|+21.82|[+20.42, +23.07]|408|1501|
|XNLI EN|Base → full_feature|+21.10|[+19.92, +22.26]|77|1134|
|XNLI EN|Base → ce_replay|-13.97|[-15.09, -12.89]|802|102|
|XNLI EN|all_layer → full_feature|-0.72|[-1.72, +0.28]|432|396|
|XNLI EN|ce_replay → full_feature|+35.07|[+33.93, +36.27]|98|1855|
|HumanEval+|Base → all_layer|-17.68|[-24.39, -10.98]|34|5|
|HumanEval+|Base → full_feature|-3.66|[-7.93, +1.22]|11|5|
|HumanEval+|Base → ce_replay|-15.24|[-23.17, -7.93]|34|9|
|HumanEval+|all_layer → full_feature|+14.02|[+7.93, +20.12]|4|27|
|HumanEval+|ce_replay → full_feature|+11.59|[+4.86, +18.90]|9|28|

## 完整材料

- [新增绝对分数](benchmarks/new_scores.csv)、[全部配对区间](benchmarks/paired_changes.csv)、[XNLI类别与选项分布](benchmarks/xnli_label_bias.json)、[代码执行审计](benchmarks/code_execution_audit.json)。
- 旧任务配对结果及57学科明细保持在此前[P2目录](../functional_representation_20261109/p2/results.md)；没有覆盖历史数据。
- 翻译本轮未补测；新增推断/代码成绩不能证明翻译退化已修复。
