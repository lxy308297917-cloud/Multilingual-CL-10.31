# 功能敏感表示诊断：探索性独立检验

本阶段未进行大模型训练。方向只由build样本构建，线性解释器只在build拟合；check保持独立。此分析受已暴露的实验A启发，非预注册；规则在表示产物生成后、行为汇总和预测结果审查前固定。

|能力|预测器|检验CE MSE|检验R²|错误行为AUC|
|---|---|---:|---:|---:|
|math|norm_total|0.382737|0.0408776|0.3619091|
|math|norm_total_functional|0.370009|0.0727733|0.3783595|
|math|norm_total_random|0.389097|0.0249399|0.3354958|
|instruction|norm_total|0.157685|-0.001375|0.7302989|
|instruction|norm_total_functional|0.159227|-0.011163|0.3308423|
|instruction|norm_total_random|0.109147|0.3068648|0.6209239|

## 额外解释力与不确定性

|能力|对照MSE减功能敏感MSE|差值|配对95%区间|
|---|---|---:|---|
|math|norm_total_minus_functional|+0.012728|[-0.009814,+0.033373]|
|math|norm_total_random_minus_functional|+0.019088|[-0.005400,+0.041070]|
|instruction|norm_total_minus_functional|-0.001541|[-0.006482,+0.003808]|
|instruction|norm_total_random_minus_functional|-0.050080|[-0.082771,-0.018728]|

正差值表示功能敏感预测器的检验误差较小。区间按64个题目ID聚类，每次保留其三个模型变体；固定build拟合系数，未包含训练种子或拟合样本不确定性，未做多重校正。

## 阶段决策

满足预先记录的探索性训练投入条件：False

该条件仅决定是否值得投入下一轮训练，不代表新方法有效。校准任务、CE和隐藏状态指标不替代正式阅读、摘要、数学或指令测评。所有负结果均保留。

数学来自可见的GSM8K训练题；指令模板范围窄且build/check共享模板。参考回答CE不是完整功能损伤，敏感方向也可能包含有益变化。原始投影量为无符号平方，不能直接区分改善与损伤；标准化系数符号和完整线性拟合保存在CSV/JSON，即使提供额外信息也不能据此宣称该方向必然有害。Base在同一LoRA-enabled引擎中复测，任何差异都不重选样本。正式测试不用于调整方向、rank、层数或系数。

完整行为表见calibration_behavior.csv，固定规范见analysis_plan.json；真实大模型参数更新为0。
