# sib200：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 73.09 | 70.43 | 69.60 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 46.93 | 61.49 | 22.06 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 70.62 | 69.78 | 60.43 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 61.00 | 60.07 | 49.72 |
| reading:acc:none | 900 | 60.89 | 67.00 | 78.56 | 69.33 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -3.49 [-5.61, -1.36] | -3.56 [-5.84, -1.29] |
| gsm8k:exact_match:strict-match | -24.87 [-28.13, -21.53] | -22.74 [-26.16, -19.56] |
| ifeval:inst_level_strict_acc:none | -10.19 [-13.27, -6.84] | -14.51 [-18.04, -10.84] |
| ifeval:prompt_level_strict_acc:none | -11.28 [-15.53, -7.02] | -16.64 [-21.07, -12.20] |
| reading:acc:none | +2.33 [-0.67, +5.11] | +8.44 [+4.78, +12.00] |
