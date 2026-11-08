# 功能敏感表示：P1综合诊断

满足进一步训练投入条件：**False**。此项只判断是否值得继续验证，不代表方法有效。

保留最初的总漂移/参数范数/随机方向比较，并额外共同控制适配前CE、提示长度和监督长度，检查额外信息是否只是样本难度或长度。补充规则在首次统计执行和行为汇总审查前固定，全部分析为探索性，非预注册。

|能力|混杂控制后的预测器|检验CE MSE|错误行为AUC|
|---|---|---:|---:|
|math|total|0.021706|0.5275718|
|math|functional|0.020369|0.5287303|
|math|random|0.021798|0.5278035|
|instruction|total|0.005304|0.5611413|
|instruction|functional|0.004990|0.6304347|
|instruction|random|0.007416|0.5842391|

## 正负证据与决定

- math：原始探索门槛=False；混杂控制门槛=False；合并支持=False。
- instruction：原始探索门槛=False；混杂控制门槛=False；合并支持=False。

原始完整结果见mechanism_diagnosis.md、diagnosis_predictors.csv、diagnosis_comparisons.csv；新增完整结果见nuisance_predictors.csv、nuisance_comparisons.csv。校准实际正确率见calibration_behavior.csv。

按64个题目ID配对bootstrap，保留同题三个模型，条件于固定拟合和单训练seed，未做多重校正。方向构造、线性拟合都只用build；check不参与构建或训练。

数学训练题可能在预训练中出现；指令只有8类共享模板，Base正确筛选改变了样本分布。平方漂移不区分有益和有害变化，CE与行为不是同一个指标。不能把本次代理解释力直接等同于正式benchmark收益，也不能由阴性结果断言此方法永远无效。

若合并条件不满足，暂不启动复杂表示约束训练。若满足，仍需审查全部反例和成本，完成固定预算四组训练和正式端点评测，才能判断方法价值。动态预算和顺序CL仍未启动。

## 评分解释补充

数学校准错误混有格式及数字提取失败，原标签未改动，详见[事后格式审查](answer_format_audit.md)。复杂约束暂缓不等于否定普通表示保持；必要实际训练对照的后续方案见[next_training_decision.md](next_training_decision.md)。
