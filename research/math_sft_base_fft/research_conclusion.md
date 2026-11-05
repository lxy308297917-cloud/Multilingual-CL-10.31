# 三语言数学FFT研究结论

路线A在两个固定epoch均未达到预先冻结门槛；已授权的条件路线B尚待独立数据/资源审计和执行，整个研究目标仍未完成。同一epoch通过语言数：{'1': 1, '2': 1}。

|语言|模型|MGSM ACC%|MSVAMP ACC%|PolyMath DW-ACC%|EN GSM8K%|EN MMLU%|IFEval strict%|HumanEval+%|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|sw|Base|9.20|26.60|3.60|73.16|57.01|66.36|65.85|
|sw|FFT-E1|25.20|49.30|1.10|58.68|56.59|51.76|54.88|
|sw|FFT-E2|33.60|49.10|1.20|59.74|55.21|52.87|56.71|
|bn|Base|47.20|60.50|4.40|73.16|57.01|66.36|65.85|
|bn|FFT-E1|39.20|48.30|0.70|59.82|57.51|53.60|57.32|
|bn|FFT-E2|46.80|51.00|0.70|62.40|56.59|54.71|55.49|
|ja|Base|62.40|76.40|7.30|73.16|57.01|66.36|65.85|
|ja|FFT-E1|46.80|60.50|1.30|64.29|57.43|51.20|53.66|
|ja|FFT-E2|54.40|61.40|1.70|64.82|57.22|51.94|51.22|

## 第一轮到第二轮具体发生了什么

### SW

E1相对Base：MGSM +16.00个百分点，MSVAMP +22.70个百分点。

E2相对E1：MGSM +8.40个百分点，MSVAMP -0.20个百分点，PolyMath DW-ACC +0.10个百分点。

E2相对E1的通用指标点估计：GSM8K +1.06个百分点；MMLU -1.39个百分点；IFEval strict +1.11个百分点；HumanEval+ +1.83个百分点。是否有明确变化以[E2−E1逐题配对区间](epoch2_minus_epoch1.md)为准，不能把每个小幅正负差值都解释为可靠改善或遗忘。

### BN

E1相对Base：MGSM -8.00个百分点，MSVAMP -12.20个百分点。

E2相对E1：MGSM +7.60个百分点，MSVAMP +2.70个百分点，PolyMath DW-ACC +0.00个百分点。

E2相对E1的通用指标点估计：GSM8K +2.58个百分点；MMLU -0.92个百分点；IFEval strict +1.11个百分点；HumanEval+ -1.83个百分点。是否有明确变化以[E2−E1逐题配对区间](epoch2_minus_epoch1.md)为准，不能把每个小幅正负差值都解释为可靠改善或遗忘。

### JA

E1相对Base：MGSM -15.60个百分点，MSVAMP -15.90个百分点。

E2相对E1：MGSM +7.60个百分点，MSVAMP +0.90个百分点，PolyMath DW-ACC +0.40个百分点。

E2相对E1的通用指标点估计：GSM8K +0.53个百分点；MMLU -0.21个百分点；IFEval strict +0.74个百分点；HumanEval+ -2.44个百分点。是否有明确变化以[E2−E1逐题配对区间](epoch2_minus_epoch1.md)为准，不能把每个小幅正负差值都解释为可靠改善或遗忘。

各任务分别判断第二轮的收益与代价，没有定义跨任务综合分数。当前只能描述两个固定预算节点；不能从单条训练轨迹确立最优epoch，也不因某项下降删除第二轮。GSM8K strict的改善可能来自输出格式变化，数学保持以flexible正确率为主；实际解析率与MMLU读出异常核验单独保留。

## 目标语言学习与通用能力变化

MGSM/MSVAMP使用最终数值ACC；PolyMath为官方表达式等价与官方难度权重；英语GSM8K使用flexible数值ACC，IFEval为官方prompt strict，代码为实际隔离执行HumanEval+ pass@1。不能把格式strict分数当作主要数学能力分数。完整正确数、其他strict/loose指标及每难度成绩见机器表。

下表仅摘出通用指标配对95%区间整体低于零的节点，其他全部正负变化保留在general_paired_comparisons，不把微小下降自动解释为稳健遗忘：

|语言|节点|通用任务|相对Base变化（百分点）|逐题配对95%区间|
|---|---|---|---:|---|
|sw|E1|GSM8K_flexible|-14.48|[-17.21, -11.68]|
|sw|E1|IFEval_prompt_strict|-14.60|[-18.85, -10.35]|
|sw|E1|HumanEval_plus|-10.98|[-17.68, -4.27]|
|sw|E2|GSM8K_flexible|-13.42|[-16.22, -10.54]|
|sw|E2|MMLU|-1.80|[-2.42, -1.18]|
|sw|E2|IFEval_prompt_strict|-13.49|[-17.93, -9.06]|
|sw|E2|HumanEval_plus|-9.15|[-15.24, -3.05]|
|bn|E1|GSM8K_flexible|-13.34|[-16.22, -10.46]|
|bn|E1|IFEval_prompt_strict|-12.75|[-17.19, -8.32]|
|bn|E1|HumanEval_plus|-8.54|[-15.24, -1.83]|
|bn|E2|GSM8K_flexible|-10.77|[-13.50, -7.96]|
|bn|E2|IFEval_prompt_strict|-11.65|[-15.90, -7.39]|
|bn|E2|HumanEval_plus|-10.37|[-16.46, -4.88]|
|ja|E1|GSM8K_flexible|-8.87|[-11.60, -6.14]|
|ja|E1|IFEval_prompt_strict|-15.16|[-19.78, -10.72]|
|ja|E1|HumanEval_plus|-12.20|[-19.51, -4.88]|
|ja|E2|GSM8K_flexible|-8.34|[-11.14, -5.53]|
|ja|E2|IFEval_prompt_strict|-14.42|[-19.04, -9.98]|
|ja|E2|HumanEval_plus|-14.63|[-21.34, -7.93]|

## 判断范围

三个语言各自从同一官方Qwen3-1.7B初始化，模型称Base指未改动的官方已后训练模型，不是另一个Qwen3-Base预训练权重。此处没有顺序语言训练，因此不能用这些结果直接证明跨语言顺序干扰或旧语言遗忘；这里观察的是单语言数学SFT之后的目标数学收益与英语通用能力变化。没有开发My Method。

九语言选取及路线决定使用了已暴露benchmark，是探索性研究。仅seed42；配对区间描述固定模型的题目抽样不确定性，未校正多重比较，不能替代训练种子稳定性或独立盲测。完整E1/E2均报告，未用英语下降早停，也未逐语言挑选最优epoch来凑门槛。PolyMath高难度存在明显长度上限命中，解析率/截断率与完整输出长度另列，按原冻结预算解释结果，不临时扩大解码预算。

## 可复核交付

- [三语言完整节点及三个预算差值](mathoctopus_epoch_comparison.md)、[完整数学成绩](target_math_results.md)、[通用能力成绩](general_retention_results.md)、[E2−E1配对区间](epoch2_minus_epoch1.md)、[路线决定](route_selection_decision.md)。
- [机器可读所有节点](../../results/math_sft_base_fft/routeA-analysis-v1/all_roles.csv)、[数学逐题配对变化](../../results/math_sft_base_fft/routeA-analysis-v1/paired_comparisons.csv)、[通用逐题配对变化](../../results/math_sft_base_fft/routeA-analysis-v1/general_paired_comparisons.csv)。
- [训练收益曲线](../../results/math_sft_base_fft/routeA-analysis-v1/figures/target_gain_vs_training.png)、[通用能力变化曲线](../../results/math_sft_base_fft/routeA-analysis-v1/figures/general_retention_vs_training.png)。曲线仅有Base/E1/E2三个真实节点，不冒充密集学习曲线。
- [成本](cost_report.md)与[真实成本记录](../../results/math_sft_base_fft/delivery-evidence-v1/cost_accounting.json)。
- [真实输入元数据与指纹](../../results/math_sft_base_fft/delivery-evidence-v1/data_inputs/manifest.json)、[模型/协议/提示/代码的实际运行身份](../../results/math_sft_base_fft/delivery-evidence-v1/actual_run_manifests/manifest.json)。原始元数据哈希与公开路径适配后的哈希分别记录。
- [输出语言辅助诊断](output_language_diagnostics.md)：固定配对样本的自动语言识别，仅作辅助，不等同于人工确认的语言正确率。
- [无答案逐题结果清单](../../results/math_sft_base_fft/delivery-evidence-v1/per_item_flags/manifest.json)：按模型与任务保存正确性、提示身份哈希及目标生成诊断，不上传原始题目、答案或生成程序。完整原始输出本地保留，可追溯。

训练器的实际loss、学习率、grad_norm日志及每GPU的PyTorch分配/预留峰值已经导出；参数更新L2仅对验证记录中注明的三个矩阵/向量采样，不冒称全模型更新范数。这些优化诊断不替代正式benchmark。

公开代码是带明确路径配置的实际运行源码整理版，部署路径适配前后哈希单独记录；九语言合格训练数据已经用公开代码真实重建并与原输入逐文件SHA一致。新主机仍需配置环境、模型与原固定评测资源，并重新通过实际代码执行隔离验证，不能把源码或复制的旧证明当作新主机安全验证。
