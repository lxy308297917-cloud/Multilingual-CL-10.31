# 三语言数学 FFT：阶段结果与交接

截至 2026-11-05T17:19:03+08:00。**这是阶段报告：三语言训练及全部目标数学评分已完成，JA的通用测评尚在运行；备用路线B尚未训练。**

## 实验设置

Qwen3-1.7B 官方原始模型（项目称Base，属于已后训练版本），先做九语言普查，冻结选择SW/BN/JA；三者各自从Base独立做MathOctopus完整两轮FFT，并非SW→BN→JA顺序训练。每种语言保留第一轮E1和第二轮E2，连续同一优化器和学习率调度，不因英语下降早停。SW/BN/JA每轮有效原始样本分别6915/6058/7024；seed42、LR5e-6、有效batch16、长度2048、BF16、assistant-only、non-thinking、不packing。

## 全部节点实际分数

全部以百分数报告；PENDING表示未完成，不能视作零分。

|训练语言|节点|MGSM-Rev2|MSVAMP|PolyMath DW-ACC|EN GSM8K|EN MMLU|IFEval|HumanEval+|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|SW|Base|9.20|26.60|3.60|73.16|57.01|66.36|65.85|
|SW|FFT E1|25.20|49.30|1.10|58.68|56.59|51.76|54.88|
|SW|FFT E2|33.60|49.10|1.20|59.74|55.21|52.87|56.71|
|BN|Base|47.20|60.50|4.40|73.16|57.01|66.36|65.85|
|BN|FFT E1|39.20|48.30|0.70|59.82|57.51|53.60|57.32|
|BN|FFT E2|46.80|51.00|0.70|62.40|56.59|54.71|55.49|
|JA|Base|62.40|76.40|7.30|73.16|57.01|66.36|65.85|
|JA|FFT E1|46.80|60.50|1.30|PENDING|PENDING|PENDING|PENDING|
|JA|FFT E2|54.40|61.40|1.70|PENDING|PENDING|PENDING|PENDING|

MGSM每语言250题，MSVAMP1000题；PolyMath每语言500题，按四难度官方表达式等价判分和DW加权。GSM8K为1319题、5-shot、灵活数值提取；MMLU为14042题、5-shot、选项log-likelihood ACC；IFEval为541条官方prompt-level strict；HumanEval+为164题、greedy pass@1、真实隔离执行。Base与训练节点使用相同题目、提示和评分协议；代码Base使用本轮重新执行的108/164，未混用旧协议结果。

## 当前结论

1. **SW已出现目标收益与通用退化共存。** E2的MGSM由9.20到33.60（+24.40个百分点），MSVAMP由26.60到49.10（+22.50）。与此同时，EN GSM8K由73.16到59.74，IFEval由66.36到52.87，HumanEval+由65.85到56.71。
2. **此现象没有覆盖三种语言。** BN E2的MGSM为46.80（Base47.20），MSVAMP51.00（Base60.50）；JA分别54.40（Base62.40）、61.40（Base76.40）。不能将BN/JA的目标退化包装成学会新语言。
3. **更难的PolyMath在三种语言均未获得收益。** 高截断率、boxed答案解析减少提示输出与推理行为变化，但这些相关性尚不能证明唯一原因，也不能证明延长生成即可恢复。
4. **冻结路线A门槛未通过。** E1/E2两个固定预算各自均只有1/3语言达标，未达到至少2/3；保留全部负结果，不重新挑语言或最佳epoch。按预定条件继续mCoT-MATH备用路线B。
5. 当前仅seed42、先前暴露benchmark下的探索结果，不能宣称跨种子稳定性，也不能据此认定特定参数干扰机制。

## 尚在进行的工作

- JA E1/E2通用能力测评及实际代码执行、随后全量配对统计与成本报告。
- 路线B：已触发mCoT-MATH数据准备；原始源题映射、抽样及token审计仍未完成，尚未开始训练。其中MathInstruct一次下载返回短文件，哈希检查已拦截，失败文件保留，不作为训练输入；另一源MetaMathQA已完整验证。
- 路线B预案为同样三语言各30k合格样本、完整两轮，须在来源去重与存储检查完成后冻结实际输入。此报告不宣称B已完成。

## 可追溯入口

- [冻结协议](protocol.md)
- [九语言Base普查](base_language_screening.md)
- [MathOctopus数据审计](mathoctopus_data_audit.md)
- [选定语言](selected_languages.md)
- [A门槛逐语言证据](../../results/math_sft_base_fft/routeA-math-gate-v1/decision.json)
- [本表机器结果与原始文件哈希](../../results/math_sft_base_fft/stage-snapshot-20261105/scores.json)

公开包包含当前实现、冻结科学配置、数据来源锁与分数；大型模型权重、原始benchmark题面/答案、生成程序和私有服务器部署状态不放入Git仓库。公开源码的路径适配及原始/公开哈希对照见source_snapshot_identity。
