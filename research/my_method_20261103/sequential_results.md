# Base→SW→Hausa：FFT与SST-NoRefresh固定阶段正式结果

更新：2026-11-03 06:00。Base/SW阶段的Hausa参考及FFT第二阶段全部6组正式任务已完成。每阶段一epoch/6400样本/400步；HA与SW来源配额相同但token/语义不匹配。Stage2 fresh Adam/scheduler。固定Final，无EN早停或正式test选模。

## 完整固定任务表



| Benchmark | Base | SW FFT Final | SW→HA FFT Final |
|---|---:|---:|---:|
| HA Belebele ACC | 23.89 | 28.44 | 26.89 |
| HA Global-MMLU ACC | 26.25 | 29.72 | 27.65 |
| SW Belebele ACC | 28.33 | 36.67 | 28.56 |
| SW Global-MMLU ACC | 27.87 | 31.65 | 27.47 |
| SW MGSM strict | 1.60 | 4.80 | 1.60 |
| SW MGSM flexible | 10.00 | 8.40 | 2.80 |
| SW XIFEval prompt strict | 14.92 | 14.69 | 13.99 |
| EN Belebele ACC | 60.78 | 77.11 | 69.11 |
| EN MMLU ACC | 57.01 | 58.45 | 56.89 |
| EN IFEval prompt strict | 66.91 | 27.17 | 25.88 |
| EN GSM8K strict | 45.03 | 3.79 | 0.15 |
| EN GSM8K flexible | 73.01 | 61.26 | 55.88 |


## 阶段2学习与遗忘必须用正确参照

定义G_HA=S_HA(W2)−S_HA(W1)，F_SW=S_SW(W1)−S_SW(W2)。

| 指标 | 变化 | 95%配对区间 |
|---|---:|---:|
| HA阅读G_HA |−1.56pp|[−5.11,+1.89]|
| HA知识G_HA |−2.07pp|[−2.89,−1.25]|
| SW阅读F_SW |+8.11pp|[+4.78,+11.44]|
| SW知识F_SW |+4.18pp|[+3.36,+5.02]|
| EN阅读W2−W1 |−8.00pp|[−10.11,−6.00]|
| EN MMLU W2−W1 |−1.56pp|[−2.03,−1.08]|
| EN GSM8K flexible W2−W1 |−5.38pp|[−8.04,−2.88]|

区间为2000次seed42配对prompt bootstrap，原doc/prompt/options逐项相同；IFEval保持提示内部指令cluster。21项完整比较保留，未校正多重比较，不选模型。

HA阶段前已经因SW适配获得跨语言增益：HA阅读Base23.89→SW28.44，知识26.25→29.72。不能把最终HA相对Base的正差当作第二阶段收益。当前固定Final的两项HA点估计都未提升，知识还出现明确负差，因此不是理想的新语言学习—旧语言遗忘正对照。

这没有证明HA数据错误，也不能分离数据质量、任务分布、预算、初始状态和负向前向迁移。开发114题的HA24.56→21.93也没有正对照；下一轮先复用现有中间checkpoint画冻结开发ACC曲线，再考虑同预算Base→HA-only。它们本轮尚未执行，不能追加正式test选最佳节点。

## 与几何和SST的关系

目标矩阵阶段累计更新cosine0.00521，而SW仍遗忘8.11pp，说明整体几何正交不蕴含功能保护。NoRefresh的顺序阶段按原计划接续，R10阶段2未运行；不拿不同初始化的绝对遗忘量忽略第一阶段目标收益差异。

当前主比较只含预冻结阅读/知识及核心数学/指令任务。第二阶段摘要、翻译及代码未重复执行，不将它们填0，也不把本表声称成所有下游任务已测。

## SST-NoRefresh第二阶段完整Final（10:15已复核）

6400条、400步、10532.56秒；源SW权重未变，两rank各3200条实际曝光的全部键和值逐项等于FFT第二阶段。W1新锚/fresh SVD与Adam/scheduler，全部old4/HA2正式任务完成。以下比较始终使用SST自己的SW阶段参考，不拿FFT W1代替。

| Benchmark | Base | SW NoRefresh Final | SW→HA NoRefresh Final |
|---|---:|---:|---:|
| HA Belebele ACC |23.89|28.67|26.89|
| HA Global-MMLU ACC |26.25|27.00|26.77|
| SW Belebele ACC |28.33|29.89|31.11|
| SW Global-MMLU ACC |27.87|27.04|28.69|
| EN Belebele ACC |60.78|77.33|77.67|
| EN MMLU ACC |57.01|57.39|56.53|
| EN IFEval prompt strict |66.91|60.26|56.01|
| EN GSM8K strict |45.03|47.23|21.08|
| EN GSM8K flexible |73.01|69.14|67.25|
| SW MGSM strict |1.60|5.20|5.20|
| SW MGSM flexible |10.00|8.40|9.60|
| SW XIFEval prompt strict |14.92|18.18|15.62|

| W2−W1 | 变化pp | 95%配对区间 |
|---|---:|---:|
| HA阅读 G_HA |−1.78|[−6.00,+2.56]|
| HA知识 G_HA |−0.23|[−1.26,+0.81]|
| SW阅读（−F_SW）|+1.22|[−2.33,+4.78]|
| SW知识（−F_SW）|+1.65|[+0.80,+2.51]|
| EN MMLU |−0.86|[−1.32,−0.42]|
| EN IFEval prompt strict |−4.25|[−7.58,−0.74]|
| EN GSM8K strict |−26.16|[−28.96,−23.35]|
| EN GSM8K flexible |−1.90|[−4.25,+0.23]|

全部6组/21项配对比较的实际doc/prompt/options一致。当前HA两项点估计仍未提高，区间跨0；不能将其相对FFT较小的HA负差解释成已解决新语言学习。SW知识在阶段2提高，但W1原本目标收益较弱，不能把不同起点的遗忘绝对差直接当作同学习水平的保护优势。数学strict大幅下降而flexible小幅变化，说明两种官方读数须共同报告；本轮未额外完成这一差异的逐题归因，不能直接声称全是格式或全是推理错误。

FFT与NoRefresh都尚未提供新HA学习正对照。R10第二阶段NOT_RUN；其SW阶段正式测评仍须完成。第二阶段摘要、翻译、代码均未测，不填0。
