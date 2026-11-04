# mtop：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 69.67 | 66.79 | 69.45 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 40.18 | 27.52 | 24.34 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 69.66 | 65.95 | 65.23 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 60.26 | 55.64 | 54.53 |
| reading:acc:none | 900 | 60.89 | 65.00 | 60.33 | 71.00 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -0.23 [-2.35, +2.05] | -3.71 [-6.29, -1.36] |
| gsm8k:exact_match:strict-match | -15.85 [-18.50, -13.04] | -20.47 [-23.73, -16.98] |
| ifeval:inst_level_strict_acc:none | -4.44 [-7.38, -1.67] | -9.71 [-13.44, -6.22] |
| ifeval:prompt_level_strict_acc:none | -5.73 [-9.43, -2.40] | -11.83 [-16.64, -7.39] |
| reading:acc:none | +6.00 [+4.00, +8.00] | +10.11 [+7.33, +12.89] |
