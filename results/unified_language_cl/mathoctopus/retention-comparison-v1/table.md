# mathoctopus：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 62.62 | 59.97 | 57.39 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 62.47 | 60.05 | 57.24 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 53.48 | 48.80 | 49.64 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 39.74 | 34.94 | 35.49 |
| reading:acc:none | 900 | 60.89 | 42.67 | 27.44 | 45.33 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -5.23 [-7.81, -2.58] | -15.77 [-18.65, -12.81] |
| gsm8k:exact_match:strict-match | -5.23 [-7.81, -2.58] | +12.43 [+9.17, +15.70] |
| ifeval:inst_level_strict_acc:none | -3.84 [-6.73, -1.07] | -25.30 [-29.19, -21.33] |
| ifeval:prompt_level_strict_acc:none | -4.25 [-7.58, -0.74] | -30.87 [-35.67, -25.88] |
| reading:acc:none | +2.67 [+1.22, +4.11] | -15.56 [-19.22, -11.67] |
