# xlsum：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 39.42 | 52.08 | 35.78 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 11.14 | 1.67 | 0.38 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 23.50 | 40.05 | 25.66 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 11.46 | 26.25 | 12.94 |
| reading:acc:none | 900 | 60.89 | 22.78 | 24.33 | 22.78 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -3.64 [-6.29, -0.83] | -37.38 [-40.33, -34.42] |
| gsm8k:exact_match:strict-match | -10.77 [-12.51, -9.17] | -44.43 [-47.08, -41.77] |
| ifeval:inst_level_strict_acc:none | +2.16 [+0.00, +4.29] | -49.28 [-53.28, -45.26] |
| ifeval:prompt_level_strict_acc:none | +1.48 [-0.55, +3.51] | -53.42 [-58.23, -48.42] |
| reading:acc:none | +0.00 [-0.56, +0.56] | -38.11 [-41.56, -34.33] |
