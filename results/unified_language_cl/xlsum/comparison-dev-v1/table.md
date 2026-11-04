# xlsum：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| xlsum | en | 256 | 10.45 | 24.44 | 20.04 | 22.99 |
| xlsum | es | 256 | 11.77 | 21.75 | 19.73 | 21.80 |
| xlsum | fr | 256 | 15.03 | 23.70 | 22.13 | 23.61 |
| xlsum | th | 256 | 18.54 | 16.49 | 25.73 | 25.85 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:xlsum | ES阶段迁移 | +13.99 [+12.77, +15.31] |
| en:xlsum | TH阶段变化 | -1.45 [-2.44, -0.48] |
| es:xlsum | G_ES | +9.98 [+8.57, +11.32] |
| es:xlsum | F_ES | -0.05 [-1.04, +0.91] |
| fr:xlsum | ES阶段迁移 | +8.67 [+7.27, +10.06] |
| fr:xlsum | TH阶段变化 | -0.09 [-1.39, +1.17] |
| th:xlsum | Single TH收益 | +7.19 [+6.20, +8.24] |
| th:xlsum | G_TH | +9.36 [+8.38, +10.39] |
| th:xlsum | T_TH | -2.04 [-2.79, -1.31] |
| th:xlsum | SEQ Final−Single TH Final | +0.12 [-0.61, +0.86] |
| th:xlsum | 阶段收益差（非因果） | +2.16 [+1.17, +3.24] |
