# wikilingua：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 56.71 | 58.00 | 35.94 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 6.14 | 0.00 | 0.00 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 49.52 | 46.76 | 37.65 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 36.60 | 32.35 | 25.14 |
| reading:acc:none | 900 | 60.89 | 36.44 | 75.11 | 28.89 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -20.77 [-23.65, -17.89] | -37.23 [-40.03, -34.12] |
| gsm8k:exact_match:strict-match | -6.14 [-7.43, -4.93] | -44.81 [-47.46, -42.15] |
| ifeval:inst_level_strict_acc:none | -11.87 [-15.06, -8.39] | -37.29 [-41.61, -32.97] |
| ifeval:prompt_level_strict_acc:none | -11.46 [-15.16, -7.58] | -41.22 [-46.40, -36.22] |
| reading:acc:none | -7.56 [-11.00, -4.22] | -32.00 [-36.33, -27.78] |
