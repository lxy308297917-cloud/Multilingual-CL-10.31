# wikilingua：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| wikilingua | en | 27004 | 18.25 | 33.05 | 31.91 | 32.89 |
| wikilingua | es | 21164 | 17.34 | 30.14 | 27.29 | 28.85 |
| wikilingua | fr | 12128 | 17.73 | 27.29 | 27.18 | 27.30 |
| wikilingua | th | 2723 | 21.01 | 16.12 | 31.52 | 32.99 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:wikilingua | ES阶段迁移 | +14.80 [+14.63, +14.98] |
| en:wikilingua | TH阶段变化 | -0.17 [-0.29, -0.05] |
| es:wikilingua | G_ES | +12.81 [+12.61, +13.00] |
| es:wikilingua | F_ES | +1.30 [+1.16, +1.43] |
| fr:wikilingua | ES阶段迁移 | +9.56 [+9.29, +9.82] |
| fr:wikilingua | TH阶段变化 | +0.01 [-0.18, +0.20] |
| th:wikilingua | Single TH收益 | +10.51 [+9.90, +11.15] |
| th:wikilingua | G_TH | +16.86 [+16.23, +17.51] |
| th:wikilingua | T_TH | -4.89 [-5.29, -4.49] |
| th:wikilingua | SEQ Final−Single TH Final | +1.46 [+1.03, +1.91] |
| th:wikilingua | 阶段收益差（非因果） | +6.35 [+5.77, +6.96] |
