# wikilingua：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| wikilingua | en | 256 | 17.49 | 33.17 | 31.05 | 32.38 |
| wikilingua | es | 256 | 16.62 | 29.78 | 26.96 | 28.58 |
| wikilingua | fr | 256 | 17.53 | 26.70 | 27.07 | 26.86 |
| wikilingua | th | 256 | 20.58 | 14.85 | 30.38 | 31.62 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:wikilingua | ES阶段迁移 | +15.68 [+13.95, +17.35] |
| en:wikilingua | TH阶段变化 | -0.78 [-1.93, +0.41] |
| es:wikilingua | G_ES | +13.16 [+11.52, +14.78] |
| es:wikilingua | F_ES | +1.20 [-0.03, +2.50] |
| fr:wikilingua | ES阶段迁移 | +9.17 [+7.55, +10.89] |
| fr:wikilingua | TH阶段变化 | +0.16 [-1.07, +1.34] |
| th:wikilingua | Single TH收益 | +9.80 [+7.94, +11.81] |
| th:wikilingua | G_TH | +16.78 [+14.72, +18.81] |
| th:wikilingua | T_TH | -5.74 [-6.82, -4.66] |
| th:wikilingua | SEQ Final−Single TH Final | +1.24 [-0.30, +2.62] |
| th:wikilingua | 阶段收益差（非因果） | +6.98 [+5.19, +8.65] |
