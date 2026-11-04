# massive：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| massive | en | 256 | 39.06 | 87.11 | 89.06 | 88.67 |
| massive | es | 256 | 28.91 | 85.94 | 85.55 | 87.11 |
| massive | fr | 256 | 17.19 | 85.16 | 84.77 | 87.50 |
| massive | th | 256 | 3.12 | 76.17 | 82.42 | 82.03 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:massive | ES阶段迁移 | +48.05 [+41.41, +54.30] |
| en:massive | TH阶段变化 | +1.56 [-1.17, +4.30] |
| es:massive | G_ES | +57.03 [+50.78, +63.67] |
| es:massive | F_ES | -1.17 [-3.52, +1.17] |
| fr:massive | ES阶段迁移 | +67.97 [+62.11, +73.83] |
| fr:massive | TH阶段变化 | +2.34 [-0.39, +5.08] |
| th:massive | Single TH收益 | +79.30 [+74.22, +84.38] |
| th:massive | G_TH | +5.86 [+1.56, +10.55] |
| th:massive | T_TH | +73.05 [+67.19, +78.52] |
| th:massive | SEQ Final−Single TH Final | -0.39 [-3.91, +3.52] |
| th:massive | 阶段收益差（非因果） | -73.44 [-80.08, -66.41] |
