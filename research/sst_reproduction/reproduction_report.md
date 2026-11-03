# SST在SW SFT上的复现与完整NoRefresh结果

更新：2026-11-03 10:20。**NoRefresh400训练与全部冻结正式测评完成；R10400仍运行，不能称两组复现已全部结束。** SST是论文baseline，不是已提出的My Method。

## 1. 实现身份和验收

作者SST仓库锁定ff0c5bfb32f621ca952d12042b046251181b8aa5；本项目在统一LLaMA-Factory/DeepSpeed3路径中独立适配实际Adam后双侧累计更新投影。并非官方trainer原样运行；R10按论文§2.3扩展。算法、恢复、环境与上游差异见[源码审计](source_audit.md)、[实现](implementation_notes.md)、[训练/测评协议](training_protocol.md)。

CPU8项数学/Adam/恢复测试、真实双rank ZeRO3 master/低精度同步与Adam矩不变测试通过。NoRefresh及R10真实Qwen连续/原生恢复全部1720574976个FP32值一致，R10全部196组参考/基亦逐值一致。公开暂存副本8项SST+4项系数测试通过。失败的非确定性与并行SVD检查保留，后者没有用于正式训练。

## 2. 固定Final正式分数

# 固定Final正式benchmark汇总

只汇入已验证完成任务；RUNNING/PENDING为尚无完整任务结果（包括排队、训练或测评中）。所有ACC为百分数，ROUGE/chrF保留原0–100尺度，不混合平均。

| Benchmark | Base | FFT | LoRA | SSU-Wanda | SST NoRefresh | SST R10 |
|---|---:|---:|---:|---:|---:|---:|
| SW Belebele ACC | 28.33 | 36.67 | 36.89 | 35.44 | 29.89 | RUNNING |
| SW Global-MMLU ACC | 27.87 | 31.65 | 31.38 | 31.24 | 27.04 | RUNNING |
| SW MGSM strict | 1.60 | 4.80 | 4.00 | 6.40 | 5.20 | RUNNING |
| SW MGSM flexible | 10.00 | 8.40 | 7.60 | 10.40 | 8.40 | RUNNING |
| SW XIFEval prompt strict | 14.92 | 14.69 | 14.92 | 14.45 | 18.18 | RUNNING |
| SW XL-Sum ROUGE-Lsum | 9.22 | 13.99 | 14.05 | 13.46 | 8.68 | RUNNING |
| EN Belebele ACC | 60.78 | 77.11 | 78.22 | 79.67 | 77.33 | RUNNING |
| EN MMLU ACC | 57.01 | 58.45 | 58.33 | 58.82 | 57.39 | RUNNING |
| EN IFEval prompt strict | 66.91 | 27.17 | 31.05 | 47.50 | 60.26 | RUNNING |
| EN GSM8K strict | 45.03 | 3.79 | 7.35 | 13.80 | 47.23 | RUNNING |
| EN GSM8K flexible | 73.01 | 61.26 | 56.79 | 65.43 | 69.14 | RUNNING |
| EN XL-Sum ROUGE-Lsum | 12.50 | 13.87 | 15.27 | 13.18 | 12.40 | RUNNING |
| EN→SW chrF++ full1012 | 5.40 | 8.00 | 7.13 | 6.13 | 5.88 | RUNNING |
| SW→EN chrF++ full1012 | 23.68 | 14.88 | 16.23 | 19.37 | 20.59 | RUNNING |


### 已核对的配对变化

NoRefresh相对Base：

- SW阅读+1.56pp，95%配对区间[-0.67,+3.67]；SW知识−0.83pp，[-1.87,+0.24]。没有显示与FFT相当的目标收益。
- EN IFEval prompt strict−6.65pp，[-10.17,−3.33]；GSM8K flexible−3.87pp，[-6.37,−1.52]，仍有旧能力退化。
- GSM8K strict+2.20pp，[-0.83,+5.16]；不能凭strict单项称数学改善。
- SW ROUGE-Lsum−0.534，[-1.078,−0.052]；EN−0.099，[-0.152,−0.046]。EN的差很小，不能把统计可检测性等同于实际巨大损失。
- EN→SW chrF++ +0.483，[+0.263,+0.707]；SW→EN−3.084，[-3.934,−2.167]。翻译按full1012 v2语料级统计，保留历史500例而不混用。

ACC按原提示/选项/逐题结果核对，实际doc和prompt完全一致。SUM/MT的ID、参考、实际prompt及token哈希逐条一致，原冻结统计代码计算区间。ACC 2000次配对prompt bootstrap，SUM/MT1000次；IFEval指令cluster不拆开；区间未做多重校正，全部描述性报告，不用于选方法。

## 3. 研究问题

1. **目标收益是否相当？** NoRefresh在阅读、知识、摘要的点估计明显弱于FFT/LoRA/SSU。R10尚未完成；不能由NoRefresh代表全部SST。
2. **是否更能保持旧能力？** 相对FFT，NoRefresh的IFEval/GSM8K保留较多；相对Base仍有退化。各英语任务不一致，不能称全面保留。
3. **是否以学习不足换保持？** 本固定配方显示这样的下游权衡，但不能从结果单独定位容量、优化或数据成因。
4. **99%是否适合？** 当前r99 NoRefresh没有达到本项目希望的目标语言增益。小自由度和训练loss只作辅助；不能看正式test后更换rank/预算并称确认。
5. **R10是否增加原Base主空间累计变化？** Final400尚待审计。中途投影器几何只作机制说明，不代替Final正式分数或完整累计变化。
6. **与SSU是否同一权衡？** NoRefresh的IFEval60.26/GSM flexible69.14较高，但SW阅读29.89/知识27.04/摘要8.68较低。SSU对应47.50/65.43及35.44/31.24/13.46；目前是不同权衡，未证明总体优势或相同目标收益下更优。

## 4. 维度、实际更新与精度

196个目标矩阵坐标1409286144，固定自由维数7924644（候选范围0.562316%）；矩阵外311288832参数不受约束。计入后者的全模型线性可行位移维数319213476（18.55272%），且所有1720574976参数参与原反向/Adam。SST并非只训练0.56%参数。

Final400原生master的196矩阵自由外能量比例3.4118e−10；实际BF16导出为18.0119%，所有310个master哈希及全部导出舍入逐值核验通过。BF16数值变化元素246630201，native master变化1409164437；自由维数少并不表示变化元素稀疏。

实际导出196矩阵外占变化能量96.2131%，主要tied embedding。能量不是功能因果；另行预登记的有限恢复/随机对照已完成1710预测，主要恢复−随机差异区间跨0，无明确功能改善。114题精度控制也没有证明普遍ACC受损，因此不把几何泄漏直接解释成遗忘。

## 5. 实际耗时与资源

| 项目 | 实测 |
|---|---:|
| NoRefresh训练wrapper wall |9956.92s（2h45m57s），两GPU |
| LF原生train runtime |9910.11s |
| 400步平均wall |24.89s/步（含本次加载/保存） |
| 原Adam+投影callback平均 |12.9678s/步，不含前后向 |
| 精确初始SVD |397.41s，正式本次复用cache，单独披露 |
| R10单次196矩阵SVD工程测量 |约391s，Final总刷新成本待完成 |
| 两rank训练期最大allocated |20751933952 bytes（约19.33GiB） |
| 两rank训练期最大reserved |24419237888 bytes（约22.74GiB） |
| NoRefresh全部正式测评wall |6334.56s，单GPU |

显存计数由adapter初始化后reset开始，不能冒充冷SVD或整节点峰值。每步原Adam提案与应用位移范数、每矩阵rank/free维度、数值变化元素及泄漏均有机器可读记录（78400行更新核验）。初始零LR步骤raw=0单独记录；最终极小提案下累计投影数值修正使相对比率放大，完整绝对值保留。

R10有40次精确刷新，仅这一部分预估约4.35小时，不能用刷新间tqdm的短ETA估整轮。当前没有已验证并采用的SVD加速；不为赶时限换近似算子。SST花费较长首先来自本轮全量Adam、参数通信和高频精确SVD，不是一般1.7B微调必然需要同样时间。

## 6. 交付边界

本报告区分完成NoRefresh、运行中R10、探索开发和未实现候选。NoRefresh顺序阶段400步及全部固定old4/HA2正式测评已完成；固定HA6400/400步、新W1锚、fresh SVD/Adam/scheduler，仅07:00前前置成功时启动。更多限制见[限制](limitations.md)。完整新方法没有被实现或验证。

## 完成后的直接FFT配对比较

NoRefresh减FFT：SW阅读−6.78pp，95%配对区间[-10.78,−2.56]；SW知识−4.61pp，[-5.65,−3.63]；EN IFEval+33.09pp，[+28.65,+37.34]；GSM8K flexible+7.88pp，[+5.31,+10.39]。均核对实际doc/prompt/options一致。它明确显示当前固定配方的学习—保持权衡，而非相同目标收益下的统一改善；全部18项对比保留。
