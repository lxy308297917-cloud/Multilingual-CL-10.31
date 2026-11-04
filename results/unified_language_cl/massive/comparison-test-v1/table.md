# massive：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| massive | en | 2974 | 37.26 | 85.88 | 84.67 | 86.42 |
| massive | es | 2974 | 24.48 | 84.03 | 79.66 | 83.83 |
| massive | fr | 2974 | 14.66 | 83.46 | 81.81 | 84.90 |
| massive | th | 2974 | 4.94 | 76.60 | 80.33 | 81.34 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:massive | ES阶段迁移 | +48.62 [+46.74, +50.44] |
| en:massive | TH阶段变化 | +0.54 [-0.20, +1.31] |
| es:massive | G_ES | +59.55 [+57.70, +61.43] |
| es:massive | F_ES | +0.20 [-0.67, +1.08] |
| fr:massive | ES阶段迁移 | +68.80 [+67.11, +70.48] |
| fr:massive | TH阶段变化 | +1.45 [+0.57, +2.35] |
| th:massive | Single TH收益 | +75.39 [+73.87, +76.87] |
| th:massive | G_TH | +4.74 [+3.63, +5.88] |
| th:massive | T_TH | +71.65 [+70.11, +73.27] |
| th:massive | SEQ Final−Single TH Final | +1.01 [+0.07, +1.98] |
| th:massive | 阶段收益差（非因果） | -70.65 [-72.49, -68.83] |
