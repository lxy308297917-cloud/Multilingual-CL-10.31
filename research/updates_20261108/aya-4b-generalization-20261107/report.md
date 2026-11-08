> 归档日期：2026-11-08。文中“暂停”等状态描述对应各实验完成时；新增AR/RU实验现已完成，见[本轮交接](../chatgpt_handoff.md)。

> [三语言结果总览](overview.md)

# Qwen3-4B：阿拉伯语与俄语泛化验证

## 设计

两种语言各自从同一官方Qwen3-4B初始化，LoRA r16/alpha32/dropout.05、LR1e-4、AdamW/cosine/warmup.05、有效batch16、6400条一遍400步、seed42、assistant-only、非思考模式、BF16、2048、不packing。固定Final400，不以英语下降早停。原暂停产物保留；没有恢复状态的两组从Base重训。
各语言配额：阅读1920、原生摘要1280、WikiLingua任务640、知识640、常识640、双向翻译各640。语言间来源、长度和监督token存在差别，翻译来自OPUS100，不能把跨语言差异纯粹归因于语言本身。
新语言翻译、Global-MMLU、额外训练种子、FFT和SSU本轮NOT_RUN。独立Base→语言实验验证适配方案的迁移性，不等于顺序CL或My Method有效。

## 目标语言任务

|语言|任务／指标|样本数|Base|Final400|变化|同题配对95%区间|
|---|---|---:|---:|---:|---:|---|
|ar|Belebele target／ACC|900|46.22|78.44|+32.22|[+28.44,+36.11]|
|ar|XNLI target／ACC|5010|33.81|44.75|+10.94|[+9.88,+12.00]|
|ar|XL-Sum target／ROUGE-Lsum|4689|9.21|20.89|+11.68|[+11.36,+11.99]|
|ru|Belebele target／ACC|900|55.56|81.89|+26.33|[+22.56,+30.00]|
|ru|XNLI target／ACC|5010|37.35|61.06|+23.71|[+22.51,+24.99]|
|ru|XL-Sum target／ROUGE-Lsum|7780|10.07|18.87|+8.80|[+8.57,+9.03]|

Belebele和XNLI是选项似然ACC（%），摘要是语言原生分词与句级LCS ROUGE-Lsum；仅比较同一种语言的Base/Final，不能将不同语言的绝对ROUGE分数直接排序。区间为2000次同题配对bootstrap，仅条件于当前训练/推理seed，没有多重比较校正；不能证明跨seed稳定。测试题不用于选择配方或checkpoint。

## 通用能力

|语言|任务／指标|Base|Final400|变化|
|---|---|---:|---:|---:|
|ar|Belebele EN／ACC|62.89|87.67|+24.78|
|ar|GSM8K／strict-match|61.03|40.03|-21.00|
|ar|GSM8K／flexible-extract|84.46|75.82|-8.64|
|ar|IFEval／prompt_level_strict_acc|78.37|42.70|-35.67|
|ar|IFEval／inst_level_strict_acc|84.65|56.47|-28.18|
|ru|Belebele EN／ACC|62.89|89.00|+26.11|
|ru|GSM8K／strict-match|61.03|21.08|-39.95|
|ru|GSM8K／flexible-extract|84.46|78.39|-6.07|
|ru|IFEval／prompt_level_strict_acc|78.37|42.33|-36.04|
|ru|IFEval／inst_level_strict_acc|84.65|56.35|-28.30|

除ROUGE外均为百分比，变化为百分点。GSM8K严格格式匹配与灵活数值提取都保留，不能将严格匹配的全部下降等同于推理遗忘。IFEval采用官方strict提示级与指令级。

## SW历史参考：三语言观察

SW原混合LoRA为研究动机的历史参考，使用同一4B、6400条与400步；AR/RU为本轮独立检验。训练来源、WikiLingua模板和翻译语料并非完全相同，不把跨语言差异当作语言本身的纯因果效应，也不把本轮称为NLLB修复方案的泛化。

|语言|任务|Base|Final400|变化|
|---|---|---:|---:|---:|
|SW（历史）|Belebele|27.33|38.11|+10.78|
|SW（历史）|XL-Sum|13.71|20.24|+6.53|
|SW（历史）|XNLI|33.19|36.97|+3.77|
|ar|Belebele target|46.22|78.44|+32.22|
|ar|XNLI target|33.81|44.75|+10.94|
|ar|XL-Sum target|9.21|20.89|+11.68|
|ru|Belebele target|55.56|81.89|+26.33|
|ru|XNLI target|37.35|61.06|+23.71|
|ru|XL-Sum target|10.07|18.87|+8.80|

## 结论边界

本表完整保留两语言所有预注册任务，不按提升幅度删任务或换语言。正向变化说明本配方在对应语言/任务上具有迁移收益；无明确收益或退化同样属于结果。若旧能力未下降，就不能声称观察到遗忘。单语言收益不构成新保护方法或顺序持续学习成功的证据。

[完整数值CSV](scores.csv) · [核验记录](verification.json)

## 选择偏好与解释限制

所有选择题均直接按候选答案似然取最大值，已从逐选项分数重算ACC，未使用自由生成答案的文本抽取或外部裁判。Base存在选项偏好，提升不能全部解释为语言知识或推理能力本身的增长。

|语言／任务|Base预测A比例|Final预测A比例|
|---|---:|---:|
|ar XNLI|99.30%|80.74%|
|ru XNLI|94.31%|63.07%|

XNLI真实三类各1670题。Base的A选项偏好明显，Final缓解但未消失；这属于当前冻结协议下的结果解释限制。选项重排或校准的新增推理本轮NOT_RUN，未据此改变当前分数、提示或Final。Belebele完整混淆矩阵也保存在classification-audit.json。

## 冻结测评设置

Belebele与XNLI为0-shot、聊天模板、候选答案似然评分。XL-Sum为0-shot贪心、最长256个生成token，输入上限7168、模型上下文8192，按语言内部比较。GSM8K为5-shot、贪心、生成上限1280；IFEval为0-shot、贪心、生成上限1280。两项生成任务沿用原停止条件和官方评分实现，精确覆盖配置在configs/core中。

## 主机重启后的恢复

主机重启中断了AR通用测评及RU摘要后续流程。原进程已消失，日志停止约2026-11-07 22:40；原日志包含尾部空字节，均原样保留。Base和两个Final400权重哈希完整，无新增训练。俄语摘要复制6728条有效唯一预测到独立目录，按原严格manifest接续剩余1052条；AR/RU的GSM8K和IFEval在独立目录按同一冻结协议重新全量运行，中断前没有完整分数用于选模或选运行。恢复命令、实际曝光和额外计算量可从本节恢复说明及本地保留的原运行/恢复日志追溯。

## 本轮结论

下列结论限于单次seed42、固定预算Final400及当前任务协议；不是跨种子或所有语言的普遍性证明。

- ar：Belebele target +32.22；XNLI target +10.94；XL-Sum target +11.68。IFEval提示级strict变化 -35.67 个百分点，GSM8K灵活提取变化 -8.64 个百分点。
- ru：Belebele target +26.33；XNLI target +23.71；XL-Sum target +8.80。IFEval提示级strict变化 -36.04 个百分点，GSM8K灵活提取变化 -6.07 个百分点。

阅读和摘要等任务的跨语言收益应结合上表实际变化与配对区间判断；通用保持按各项分别报告，不能把英语阅读的提升概括为所有英语/通用能力都保持。训练的翻译来源与WikiLingua模板不同，也不能把AR/RU结果称为SW NLLB替换方案的泛化。

## 摘要补充指标

[摘要全部评分](summary-all-metrics.csv)保留ROUGE-1、ROUGE-2、ROUGE-Lsum及辅助chrF++。俄语摘要的ROUGE-Lsum提升，但摘要辅助chrF++从26.28降至25.56；不能将主要指标提升写成所有摘要指标都提升。这里的chrF++用于摘要文本比较，不是本轮新增翻译测评。主要指标始终为提前冻结的ROUGE-Lsum。
