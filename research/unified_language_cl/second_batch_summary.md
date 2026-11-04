# 阶段汇总：四轨道正式结果已齐

全项目21段主训练完成；此表覆盖已完成原生与通用能力测评的4条轨道。XL-Sum、WikiLingua、SIB-200的完整比较仍在进行，P3对照尚未运行。没有据此宣布整个目标完成。

W1=Single ES Final；W2=ES→TH Final。原生指标为ACC或结构EM；通用列均为百分数。不同任务不求综合平均。

| 原生任务/单位 | TH Base→W1→W2 | TH阶段增益与95%区间 | ES W1→W2 | ES遗忘与95%区间 | GSM8K flexible Base→W2 | IFEval prompt strict Base→W2 | EN Belebele Base→W2 |
|---|---|---|---|---|---|---|---|
| massive / ACC% | 4.94→76.60→81.34 | +4.74 [+3.63, +5.88] | 84.03→83.83 | +0.20 [-0.67, +1.08] | 73.16→67.17 | 66.36→44.18 | 60.89→79.44 |
| mgsm_rev2 / ACC% | 58.00→43.20→45.60 | +2.40 [-4.00, +8.40] | 56.00→52.80 | +3.20 [-2.80, +9.20] | 73.16→57.39 | 66.36→35.49 | 60.89→45.33 |
| msvamp / ACC% | 75.50→56.80→59.10 | +2.30 [-0.50, +5.20] | 64.60→63.80 | +0.80 [-1.80, +3.30] | 73.16→57.39 | 66.36→35.49 | 60.89→45.33 |
| mtop / Tree EM% | 0.00→12.91→49.01 | +36.09 [+34.14, +37.97] | 66.18→65.11 | +1.07 [-0.27, +2.40] | 73.16→69.45 | 66.36→54.53 | 60.89→71.00 |
| xnli / ACC% | 55.25→68.40→74.97 | +6.57 [+5.35, +7.86] | 82.08→82.53 | -0.46 [-1.22, +0.26] | 73.16→42.00 | 66.36→57.30 | 60.89→81.67 |

## 可支持的阶段判断

- MASSIVE、MTOP、XNLI出现目标任务学习，并同时在部分通用任务上下降。英语阅读往往提高，不能概括为全部英语能力遗忘。
- 已完成的这些轨道还没有提供稳定的同任务旧ES下降证据；ES遗忘区间均跨零。不能为了制造期望趋势而改预算或挑选结果。
- MathOctopus的正式数学结果低于Base，即使TH第二阶段略恢复，也不能称为成功获得目标数学能力。输出复核已确认，格式/长度因素不能单独解释大幅退化。
- 所有区间条件于seed42固定模型；没有训练种子重复，未做多重检验校正。
- 同一个MathOctopus模型在MGSM与MSVAMP两行出现，其通用能力列是同一份结果，不是两次独立测评。完整四语言、Single TH对照、strict/loose与全部区间见逐轨道文件。

## 完整来源

- [massive原生完整矩阵](../../results/unified_language_cl/massive/comparison-test-v1/table.md)；[通用完整矩阵](../../results/unified_language_cl/massive/retention-comparison-v1/table.md)
- [mathoctopus原生完整矩阵](../../results/unified_language_cl/mathoctopus/comparison-test-v1/table.md)；[通用完整矩阵](../../results/unified_language_cl/mathoctopus/retention-comparison-v1/table.md)
- [mtop原生完整矩阵](../../results/unified_language_cl/mtop/comparison-test-v1/table.md)；[通用完整矩阵](../../results/unified_language_cl/mtop/retention-comparison-v1/table.md)
- [xnli原生完整矩阵](../../results/unified_language_cl/xnli/comparison-test-v1/table.md)；[通用完整矩阵](../../results/unified_language_cl/xnli/retention-comparison-v1/table.md)
