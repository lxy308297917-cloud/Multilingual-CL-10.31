# 已完成正式任务的配对变化

| Task | Metric | Reference | Target | Δ pp | 95% paired interval |
|---|---|---:|---:|---:|---:|
| sft_v1_gsm8k | exact_match,flexible-extract | 61.26 | 69.14 | +7.88 | [+5.31, +10.39] |
| sft_v1_gsm8k | exact_match,strict-match | 3.79 | 47.23 | +43.44 | [+40.64, +46.25] |
| sft_v1_ifeval | inst_level_loose_acc,none | 41.49 | 71.82 | +30.34 | [+26.45, +34.06] |
| sft_v1_ifeval | inst_level_strict_acc,none | 41.01 | 69.18 | +28.18 | [+24.29, +31.88] |
| sft_v1_ifeval | prompt_level_loose_acc,none | 27.36 | 62.48 | +35.12 | [+30.68, +39.56] |
| sft_v1_ifeval | prompt_level_strict_acc,none | 27.17 | 60.26 | +33.09 | [+28.65, +37.34] |
| sft_v1_mgsm_native_cot_sw | exact_match,flexible-extract | 8.40 | 8.40 | +0.00 | [-4.00, +4.00] |
| sft_v1_mgsm_native_cot_sw | exact_match,strict-match | 4.80 | 5.20 | +0.40 | [-2.80, +3.60] |
| sft_v1_xifeval_sw | inst_level_loose_acc,none | 23.06 | 26.86 | +3.79 | [+0.30, +7.57] |
| sft_v1_xifeval_sw | inst_level_strict_acc,none | 23.06 | 26.38 | +3.32 | [-0.17, +7.02] |
| sft_v1_xifeval_sw | prompt_level_loose_acc,none | 14.69 | 18.18 | +3.50 | [-0.23, +7.23] |
| sft_v1_xifeval_sw | prompt_level_strict_acc,none | 14.69 | 18.18 | +3.50 | [-0.23, +7.23] |
| belebele_eng_Latn_locked | acc,none | 77.11 | 77.33 | +0.22 | [-2.22, +2.56] |
| belebele_eng_Latn_locked | acc_norm,none | 77.11 | 77.33 | +0.22 | [-2.22, +2.56] |
| belebele_swh_Latn_locked | acc,none | 36.67 | 29.89 | -6.78 | [-10.78, -2.56] |
| belebele_swh_Latn_locked | acc_norm,none | 36.67 | 29.89 | -6.78 | [-10.78, -2.56] |
| mmlu_locked | acc,none | 58.45 | 57.39 | -1.05 | [-1.66, -0.43] |
| global_mmlu_full_sw_locked | acc,none | 31.65 | 27.04 | -4.61 | [-5.65, -3.63] |

所有实际doc和prompt/options逐条相同。区间为描述性，未作多重校正，不用于挑任务或选模型。
未完成任务：
