# mathoctopus：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| mgsm_rev2 | en | 250 | 80.80 | 63.60 | 63.60 | 61.60 |
| msvamp | en | 1000 | 90.60 | 65.10 | 60.30 | 59.90 |
| mgsm_rev2 | es | 250 | 74.80 | 56.00 | 52.80 | 52.80 |
| msvamp | es | 1000 | 82.60 | 64.60 | 59.30 | 63.80 |
| mgsm_rev2 | fr | 250 | 71.60 | 54.00 | 55.60 | 51.20 |
| msvamp | fr | 1000 | 83.40 | 60.90 | 58.10 | 59.20 |
| mgsm_rev2 | th | 250 | 58.00 | 43.20 | 44.40 | 45.60 |
| msvamp | th | 1000 | 75.50 | 56.80 | 56.50 | 59.10 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:mgsm_rev2 | ES阶段迁移 | -17.20 [-24.00, -10.80] |
| en:mgsm_rev2 | TH阶段变化 | -2.00 [-8.00, +4.00] |
| en:msvamp | ES阶段迁移 | -25.50 [-28.50, -22.60] |
| en:msvamp | TH阶段变化 | -5.20 [-7.80, -2.60] |
| es:mgsm_rev2 | G_ES | -18.80 [-24.80, -12.40] |
| es:mgsm_rev2 | F_ES | +3.20 [-2.80, +9.20] |
| es:msvamp | G_ES | -18.00 [-21.10, -15.00] |
| es:msvamp | F_ES | +0.80 [-1.80, +3.30] |
| fr:mgsm_rev2 | ES阶段迁移 | -17.60 [-24.80, -10.40] |
| fr:mgsm_rev2 | TH阶段变化 | -2.80 [-8.41, +3.60] |
| fr:msvamp | ES阶段迁移 | -22.50 [-25.60, -19.50] |
| fr:msvamp | TH阶段变化 | -1.70 [-4.40, +1.00] |
| th:mgsm_rev2 | Single TH收益 | -13.60 [-20.00, -7.60] |
| th:mgsm_rev2 | G_TH | +2.40 [-4.00, +8.40] |
| th:mgsm_rev2 | T_TH | -14.80 [-21.20, -8.00] |
| th:mgsm_rev2 | SEQ Final−Single TH Final | +1.20 [-4.00, +6.40] |
| th:mgsm_rev2 | 阶段收益差（非因果） | +16.00 [+8.00, +24.40] |
| th:msvamp | Single TH收益 | -19.00 [-22.20, -15.70] |
| th:msvamp | G_TH | +2.30 [-0.50, +5.20] |
| th:msvamp | T_TH | -18.70 [-21.60, -15.50] |
| th:msvamp | SEQ Final−Single TH Final | +2.60 [+0.20, +5.20] |
| th:msvamp | 阶段收益差（非因果） | +21.30 [+17.20, +25.40] |
