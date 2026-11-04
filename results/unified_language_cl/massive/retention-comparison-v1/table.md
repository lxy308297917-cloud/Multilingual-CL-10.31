# massive：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 69.29 | 53.90 | 67.17 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 9.40 | 1.06 | 5.00 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 63.79 | 57.55 | 56.24 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 51.76 | 45.84 | 44.18 |
| reading:acc:none | 900 | 60.89 | 76.89 | 80.56 | 79.44 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -2.12 [-4.32, -0.08] | -5.99 [-8.49, -3.34] |
| gsm8k:exact_match:strict-match | -4.40 [-6.07, -2.73] | -39.80 [-42.76, -36.92] |
| ifeval:inst_level_strict_acc:none | -7.55 [-10.66, -4.47] | -18.71 [-22.65, -14.62] |
| ifeval:prompt_level_strict_acc:none | -7.58 [-11.46, -3.70] | -22.18 [-26.99, -17.19] |
| reading:acc:none | +2.56 [+0.78, +4.44] | +18.56 [+15.00, +22.11] |
