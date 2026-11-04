# xnli：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 50.57 | 66.94 | 42.00 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 6.07 | 17.06 | 4.47 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 70.86 | 68.71 | 68.47 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 61.37 | 58.23 | 57.30 |
| reading:acc:none | 900 | 60.89 | 82.00 | 80.44 | 81.67 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -8.57 [-10.92, -6.37] | -31.16 [-34.19, -28.13] |
| gsm8k:exact_match:strict-match | -1.59 [-3.11, -0.15] | -40.33 [-43.14, -37.68] |
| ifeval:inst_level_strict_acc:none | -2.40 [-5.10, +0.23] | -6.47 [-9.55, -3.66] |
| ifeval:prompt_level_strict_acc:none | -4.07 [-7.39, -1.10] | -9.06 [-13.31, -5.18] |
| reading:acc:none | -0.33 [-2.00, +1.33] | +20.78 [+17.56, +23.89] |
