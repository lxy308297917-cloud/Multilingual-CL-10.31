# xlsum：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| xlsum | en | 11535 | 10.40 | 23.81 | 20.66 | 23.03 |
| xlsum | es | 4763 | 11.88 | 20.99 | 19.44 | 21.15 |
| xlsum | fr | 1086 | 15.63 | 24.14 | 22.85 | 24.04 |
| xlsum | th | 826 | 18.62 | 16.53 | 25.60 | 25.67 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:xlsum | ES阶段迁移 | +13.41 [+13.23, +13.60] |
| en:xlsum | TH阶段变化 | -0.78 [-0.93, -0.62] |
| es:xlsum | G_ES | +9.11 [+8.82, +9.41] |
| es:xlsum | F_ES | -0.16 [-0.39, +0.06] |
| fr:xlsum | ES阶段迁移 | +8.51 [+7.86, +9.20] |
| fr:xlsum | TH阶段变化 | -0.10 [-0.66, +0.49] |
| th:xlsum | Single TH收益 | +6.99 [+6.42, +7.55] |
| th:xlsum | G_TH | +9.14 [+8.55, +9.76] |
| th:xlsum | T_TH | -2.09 [-2.52, -1.66] |
| th:xlsum | SEQ Final−Single TH Final | +0.07 [-0.32, +0.46] |
| th:xlsum | 阶段收益差（非因果） | +2.16 [+1.59, +2.75] |
