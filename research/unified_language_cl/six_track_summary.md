# 六轨道完整结果：七轨道主实验进行中

21段主训练已完成。MASSIVE、MathOctopus、MTOP、XNLI、XL-Sum、SIB-200的四语言固定Final与通用能力完整测评已齐；WikiLingua正在生成其每模型63,019条测试摘要。预定中间开发节点与三个合格轨道的Same-ID/Disjoint-ID对照仍待完成，本页不代表整个目标完成。

W1=共享Single ES Final，W2=ES→TH Final。除摘要为ROUGE-Lsum分，其他原生列为ACC或Tree EM百分数。通用列为百分数，不计算跨任务平均。

| 原生任务 | 指标 | TH Base→W1→W2 | TH阶段变化及95%区间 | ES W1→W2 | ES遗忘及95%区间 | GSM8K flexible Base→W2 | IFEval prompt strict Base→W2 | EN Belebele Base→W2 |
|---|---|---|---|---|---|---|---|---|
| massive | ACC% | 4.94→76.60→81.34 | +4.74 [+3.63, +5.88] | 84.03→83.83 | +0.20 [-0.67, +1.08] | 73.16→67.17 | 66.36→44.18 | 60.89→79.44 |
| mgsm_rev2 | ACC% | 58.00→43.20→45.60 | +2.40 [-4.00, +8.40] | 56.00→52.80 | +3.20 [-2.80, +9.20] | 73.16→57.39 | 66.36→35.49 | 60.89→45.33 |
| msvamp | ACC% | 75.50→56.80→59.10 | +2.30 [-0.50, +5.20] | 64.60→63.80 | +0.80 [-1.80, +3.30] | 73.16→57.39 | 66.36→35.49 | 60.89→45.33 |
| mtop | Tree EM% | 0.00→12.91→49.01 | +36.09 [+34.14, +37.97] | 66.18→65.11 | +1.07 [-0.27, +2.40] | 73.16→69.45 | 66.36→54.53 | 60.89→71.00 |
| xnli | ACC% | 55.25→68.40→74.97 | +6.57 [+5.35, +7.86] | 82.08→82.53 | -0.46 [-1.22, +0.26] | 73.16→42.00 | 66.36→57.30 | 60.89→81.67 |
| xlsum | ROUGE-Lsum分 | 18.62→16.53→25.67 | +9.14 [+8.55, +9.76] | 20.99→21.15 | -0.16 [-0.39, +0.06] | 73.16→35.78 | 66.36→12.94 | 60.89→22.78 |
| sib200 | ACC% | 15.20→83.82→87.25 | +3.43 [+0.00, +6.86] | 85.78→88.73 | -2.94 [-7.35, +1.47] | 73.16→69.60 | 66.36→49.72 | 60.89→69.33 |

## 当前可支持的判断

- MASSIVE、MTOP、XNLI、SIB-200与XL-Sum有匹配目标任务学习，也出现部分通用任务退化；MathOctopus是明确保留的负结果，目标数学成绩低于Base。
- 目前未观察到可靠的同任务旧西语下降，各轨道旧ES遗忘区间跨零。不能把跨任务退化改称为旧语言同任务遗忘。
- XL-Sum的TH阶段ROUGE恢复伴随长度控制修复：ES后泰语达生成上限60.90%，Final为0.85%。不能将全部改善解释为新语义知识。多数理解轨道英语阅读提高，而XL-Sum阅读下降；不能合成一个“英语全面遗忘”结论。
- 冻结开发64题的同题候选分数也改善，因此分类增益包含标签读出修复，但不能全部归为生成格式；候选评分仍包含任务和标签先验，不单独证明新语言语义知识。
- 每段预算3200例/200步（SIB640例/40步），有效batch16，seed42，LR1e-5，固定Final。SEQ累计两段预算，Single TH一段，不是同总算力。TH实际输入/标签顺序已核验一致；不因英语下降早停。
- 95%区间条件于固定seed42模型，不涵盖训练随机性，未做多重检验校正；区间跨零不等于等价。同一个MathOctopus模型的两行通用成绩来自同一份测评。
- P3仅对满足原预算的MASSIVE、XNLI、MTOP运行；三条尚未启动，不据本页结果修改预留样本、预算或学习率。

## 证据入口

- massive：[原生完整矩阵](../../results/unified_language_cl/massive/comparison-test-v1/table.md)，[通用完整矩阵](../../results/unified_language_cl/massive/retention-comparison-v1/table.md)，[原生辅助指标](../../results/unified_language_cl/massive/native-auxiliary-v1/table.md)。
- mathoctopus：[原生完整矩阵](../../results/unified_language_cl/mathoctopus/comparison-test-v1/table.md)，[通用完整矩阵](../../results/unified_language_cl/mathoctopus/retention-comparison-v1/table.md)，[原生辅助指标](../../results/unified_language_cl/mathoctopus/native-auxiliary-v1/table.md)。
- mtop：[原生完整矩阵](../../results/unified_language_cl/mtop/comparison-test-v1/table.md)，[通用完整矩阵](../../results/unified_language_cl/mtop/retention-comparison-v1/table.md)，[原生辅助指标](../../results/unified_language_cl/mtop/native-auxiliary-v1/table.md)。
- xnli：[原生完整矩阵](../../results/unified_language_cl/xnli/comparison-test-v1/table.md)，[通用完整矩阵](../../results/unified_language_cl/xnli/retention-comparison-v1/table.md)，[原生辅助指标](../../results/unified_language_cl/xnli/native-auxiliary-v1/table.md)。
- xlsum：[原生完整矩阵](../../results/unified_language_cl/xlsum/comparison-test-v1/table.md)，[通用完整矩阵](../../results/unified_language_cl/xlsum/retention-comparison-v1/table.md)，[原生辅助指标](../../results/unified_language_cl/xlsum/native-auxiliary-v1/table.md)。
- sib200：[原生完整矩阵](../../results/unified_language_cl/sib200/comparison-test-v1/table.md)，[通用完整矩阵](../../results/unified_language_cl/sib200/retention-comparison-v1/table.md)，[原生辅助指标](../../results/unified_language_cl/sib200/native-auxiliary-v1/table.md)。
- [XL-Sum完整解释](xlsum_results.md)；[SIB-200完整解释](sib200_results.md)。
- [48组开发读出诊断](../../results/unified_language_cl/classification_readout_complete_v2/table.md)；[摘要输出长度与语言诊断](../../results/unified_language_cl/xlsum/output-audit-v1/table.md)。
- [更新后的七轨道数据审计](dataset_audit.md)；[语言身份与对齐限制](language_alignment.md)；[初版数据审计快照](dataset_audit_initial_snapshot.md)。
