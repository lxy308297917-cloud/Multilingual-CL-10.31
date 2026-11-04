# mathoctopus：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| mathoctopus | en | 256 | 91.41 | 76.95 | 70.31 | 73.83 |
| mathoctopus | es | 256 | 77.34 | 65.62 | 64.06 | 61.72 |
| mathoctopus | fr | 256 | 80.47 | 64.84 | 60.94 | 64.84 |
| mathoctopus | th | 256 | 58.59 | 50.00 | 49.22 | 58.98 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:mathoctopus | ES阶段迁移 | -14.45 [-19.53, -9.77] |
| en:mathoctopus | TH阶段变化 | -3.12 [-7.81, +1.17] |
| es:mathoctopus | G_ES | -11.72 [-17.58, -6.25] |
| es:mathoctopus | F_ES | +3.91 [-1.56, +9.38] |
| fr:mathoctopus | ES阶段迁移 | -15.62 [-22.27, -9.38] |
| fr:mathoctopus | TH阶段变化 | +0.00 [-5.08, +5.08] |
| th:mathoctopus | Single TH收益 | -9.38 [-16.41, -2.73] |
| th:mathoctopus | G_TH | +8.98 [+2.34, +15.63] |
| th:mathoctopus | T_TH | -8.59 [-15.23, -1.56] |
| th:mathoctopus | SEQ Final−Single TH Final | +9.77 [+4.69, +15.23] |
| th:mathoctopus | 阶段收益差（非因果） | +18.36 [+8.98, +27.34] |
