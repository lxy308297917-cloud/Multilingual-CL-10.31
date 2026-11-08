> 归档日期：2026-11-08。文中“暂停”等状态描述对应各实验完成时；新增AR/RU实验现已完成，见[本轮交接](../chatgpt_handoff.md)。

# NLLB与三线对照：结果入口

训练与全部既定测评已完成。仅单个训练/推理seed42；未按正式测试选择checkpoint或新增训练配方。其他语言保持暂停。

- [新增NLLB完整报告](report.md)
- [原三线完整报告](../translation-three-lines-20261106/report.md)
- [双向翻译全部分数](translation-analysis-v1/scores.csv)
- [同题配对比较与区间](translation-analysis-v1/paired_comparisons.csv)
- [其他任务对比CSV](combined-retention.csv)

## 其他任务结果

|任务／指标|Base|原混合|清洗GV混合|NLLB混合|样本／指令数|
|---|---:|---:|---:|---:|---:|
|Belebele EN／ACC|62.89|86.33|87.89|85.44|900|
|Belebele SW／ACC|27.33|38.11|38.67|38.33|900|
|xnli_sw／ACC|33.19|36.97|33.95|38.32|5010|
|Summary SW／ROUGE-Lsum|13.71|20.24|20.36|20.58|987|
|GSM8K strict-match／ACC|61.03|59.74|67.78|36.69|1319|
|GSM8K flexible-extract／ACC|84.46|81.20|80.89|80.06|1319|
|IFEval／prompt_level_strict_acc|78.37|49.35|51.94|45.84|541|
|IFEval／inst_level_strict_acc|84.65|62.47|63.79|58.63|834|

除摘要ROUGE-Lsum外均为百分比。GSM8K灵活提取与严格格式评分分别保留；IFEval采用官方strict提示级/指令级指标。新的Global-MMLU、EN MMLU、EN摘要及XCOPA本轮NOT_RUN。

NLLB相对Base的SW阅读增加11.00个百分点、XNLI增加5.13个百分点、摘要增加6.87 ROUGE-Lsum；IFEval提示级strict下降32.53个百分点，GSM8K灵活提取下降4.40个百分点。GSM8K严格匹配下降24.34个百分点，但不能把严格格式评分的全部差距直接解释为数学推理遗忘；它与灵活提取存在明显差异，具体格式错误仍需另行逐例分析。本轮未扩大该诊断。

## 核验与解释

原5120条非翻译样本的内容、位置以及实际训练序列一致；1280条翻译槽位均替换。NLLB400步曝光6400条，每条一遍，翻译占20%样本但只占约10.46%监督token。
翻译两方向均1012句，数据/提示/生成后端及批量身份已核对，chrF++直接重算，配对bootstrap2000次。阅读、XNLI、GSM8K及IFEval逐题重算与身份核对通过；摘要987句使用原冻结评分协议。
NLLB没有解决EN→SW贪心重复；其SW→EN比原混合更好，但对Base无明确提升。清洗GV在EN→SW采样上存在收益，原贪心退化和反向退化仍需报告。不能据此断言噪声或梯度冲突的因果机制。
