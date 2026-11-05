# ES→TH 三角形阶段结果表

Task 1 = 西语（ES），Task 2 = 泰语（TH）。Stage 1 为 Base→ES Final；Stage 2 为共享 ES Final→TH Final。下面展示实际分数，不是分数变化。

原生任务同时测了 ES/TH/FR/EN；本表只取两个训练语言。`—` 是按三角形展示省略未训练语言，TH 在 Stage 1 的实测成绩仍保留在机器可读表和[完整四语言矩阵](stage_language_matrix.md)。GSM8K、IFEval及EN Belebele是另列的英语通用监测，不是Task 1/2的泰语版本。

| Benchmark／指标 | 训练阶段 | Task 1：西语 ES | Task 2：泰语 TH |
|---|---|---:|---:|
| **MASSIVE / ACC%** | Stage 1：ES | 84.03 | — |
| | Stage 2：ES→TH | 83.83 | 81.34 |
| **XNLI / ACC%** | Stage 1：ES | 82.08 | — |
| | Stage 2：ES→TH | 82.53 | 74.97 |
| **MTOP / Tree EM%** | Stage 1：ES | 66.18 | — |
| | Stage 2：ES→TH | 65.11 | 49.01 |
| **XL-Sum / ROUGE-Lsum** | Stage 1：ES | 20.99 | — |
| | Stage 2：ES→TH | 21.15 | 25.67 |
| **WikiLingua / ROUGE-Lsum** | Stage 1：ES | 30.14 | — |
| | Stage 2：ES→TH | 28.85 | 32.99 |
| **SIB-200 / ACC%** | Stage 1：ES | 85.78 | — |
| | Stage 2：ES→TH | 88.73 | 87.25 |
| **MathOctopus / MGSM-Rev2 / ACC%** | Stage 1：ES | 56.00 | — |
| | Stage 2：ES→TH | 52.80 | 45.60 |
| **MathOctopus / MSVAMP / ACC%** | Stage 1：ES | 64.60 | — |
| | Stage 2：ES→TH | 63.80 | 59.10 |

## 判断目标学习所需的基线

三角表中的Task 1上下比较表示旧任务保持；Task 2的Final还应与原模型和直接泰语训练比较。SEQ累计两阶段预算，直接TH只有一阶段，不能当等总算力比较。

| Benchmark | ES原模型 | TH原模型 | 直接TH Final | ES→TH的TH Final |
|---|---:|---:|---:|---:|
| MASSIVE | 24.48 | 4.94 | 80.33 | 81.34 |
| XNLI | 48.74 | 55.25 | 74.69 | 74.97 |
| MTOP | 0.13 | 0.00 | 45.68 | 49.01 |
| XL-Sum | 11.88 | 18.62 | 25.60 | 25.67 |
| WikiLingua | 17.34 | 21.01 | 31.52 | 32.99 |
| SIB-200 | 50.49 | 15.20 | 84.80 | 87.25 |
| MathOctopus / MGSM-Rev2 | 74.80 | 58.00 | 44.40 | 45.60 |
| MathOctopus / MSVAMP | 82.60 | 75.50 | 56.50 | 59.10 |

单位：分类和数字答案为ACC%，MTOP为结构树精确匹配%，摘要为ROUGE-Lsum F1分，不跨任务平均。MathOctopus两个数学任务Final均低于原模型。

WikiLingua为官方划分的唯一映射非空子集，非全量官方test。模型均固定seed42与预定Final，没有依据这些分数早停或选择checkpoint。变化的配对区间见[FFT-SEQ结果](fft_seq_results.md)。

[机器可读实际分数](../../results/unified_language_cl/triangle-tables-v1/scores.csv)保留未舍入值、样本量和Stage 1泰语实测值；同一分数的不同表格仅改变展示形式。
