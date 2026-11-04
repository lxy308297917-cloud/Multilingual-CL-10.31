# Fixed100: all native formal benchmark metrics

| Task and metric | n | W1 | A SFT | B marginal | C content | D Replay |
|---|---:|---:|---:|---:|---:|---:|
| EN Belebele / acc / none | 900 | 77.11 | 69.44 | 74.22 | 74.00 | 73.78 |
| EN Belebele / acc_norm / none | 900 | 77.11 | 69.44 | 74.22 | 74.00 | 73.78 |
| SW Belebele / acc / none | 900 | 36.67 | 28.22 | 34.11 | 32.11 | 34.78 |
| SW Belebele / acc_norm / none | 900 | 36.67 | 28.22 | 34.11 | 32.11 | 34.78 |
| SW Global-MMLU / acc / none | 14042 | 31.65 | 25.95 | 29.47 | 29.53 | 30.36 |
| EN MMLU / acc / none | 14042 | 58.45 | 55.84 | 57.89 | 58.30 | 58.45 |
| sft_v1_gsm8k / exact_match / flexible-extract | 1319 | 61.26 | 58.15 | 61.64 | 57.32 | 59.59 |
| sft_v1_gsm8k / exact_match / strict-match | 1319 | 3.79 | 0.23 | 2.96 | 0.61 | 1.67 |
| sft_v1_ifeval / prompt_level_strict_acc / none | 541 | 27.17 | 24.95 | 26.25 | 25.51 | 27.73 |
| sft_v1_ifeval / inst_level_strict_acc / none | 834 | 41.01 | 39.33 | 41.61 | 39.45 | 41.61 |
| sft_v1_ifeval / prompt_level_loose_acc / none | 541 | 27.36 | 24.95 | 26.43 | 25.51 | 27.73 |
| sft_v1_ifeval / inst_level_loose_acc / none | 834 | 41.49 | 39.33 | 41.85 | 39.57 | 41.61 |
| sft_v1_mgsm_native_cot_sw / exact_match / flexible-extract | 250 | 8.40 | 2.00 | 2.40 | 2.40 | 4.00 |
| sft_v1_mgsm_native_cot_sw / exact_match / strict-match | 250 | 4.80 | 0.40 | 0.80 | 0.40 | 2.40 |
| sft_v1_xifeval_sw / prompt_level_strict_acc / none | 429 | 14.69 | 14.45 | 14.22 | 14.92 | 14.45 |
| sft_v1_xifeval_sw / inst_level_strict_acc / none | 633 | 23.06 | 24.01 | 23.06 | 23.38 | 22.91 |
| sft_v1_xifeval_sw / prompt_level_loose_acc / none | 429 | 14.69 | 14.45 | 14.22 | 14.92 | 14.45 |
| sft_v1_xifeval_sw / inst_level_loose_acc / none | 633 | 23.06 | 24.17 | 23.38 | 23.70 | 23.06 |
| HA Belebele / acc / none | 900 | 28.44 | 24.89 | 28.56 | 27.33 | 26.78 |
| HA Belebele / acc_norm / none | 900 | 28.44 | 24.89 | 28.56 | 27.33 | 26.78 |
| HA Global-MMLU / acc / none | 14042 | 29.72 | 26.02 | 28.65 | 29.06 | 29.32 |
| en_sw / chrF++ | 1012 | 8.00 | 4.22 | 4.62 | 4.69 | 4.87 |
| en_sw / BLEU | 1012 | 0.31 | 0.08 | 0.11 | 0.11 | 0.11 |
| sw_en / chrF++ | 1012 | 14.88 | 13.46 | 14.81 | 14.61 | 15.98 |
| sw_en / BLEU | 1012 | 2.22 | 1.42 | 1.68 | 1.61 | 1.82 |

Percent for ACC and native0–100 for corpus translation scores. All metrics retained. See analysis.json for paired intervals and exact identity checks.
