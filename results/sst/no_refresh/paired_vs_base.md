# 已完成正式任务的配对变化

| Task | Metric | Reference | Target | Δ pp | 95% paired interval |
|---|---|---:|---:|---:|---:|
| sft_v1_gsm8k | exact_match,flexible-extract | 73.01 | 69.14 | -3.87 | [-6.37, -1.52] |
| sft_v1_gsm8k | exact_match,strict-match | 45.03 | 47.23 | +2.20 | [-0.83, +5.16] |
| sft_v1_ifeval | inst_level_loose_acc,none | 78.18 | 71.82 | -6.35 | [-8.99, -3.67] |
| sft_v1_ifeval | inst_level_strict_acc,none | 75.18 | 69.18 | -6.00 | [-8.56, -3.28] |
| sft_v1_ifeval | prompt_level_loose_acc,none | 70.43 | 62.48 | -7.95 | [-11.46, -4.44] |
| sft_v1_ifeval | prompt_level_strict_acc,none | 66.91 | 60.26 | -6.65 | [-10.17, -3.33] |
| sft_v1_mgsm_native_cot_sw | exact_match,flexible-extract | 10.00 | 8.40 | -1.60 | [-4.80, +1.60] |
| sft_v1_mgsm_native_cot_sw | exact_match,strict-match | 1.60 | 5.20 | +3.60 | [+1.20, +6.40] |
| sft_v1_xifeval_sw | inst_level_loose_acc,none | 25.91 | 26.86 | +0.95 | [-2.75, +4.69] |
| sft_v1_xifeval_sw | inst_level_strict_acc,none | 23.22 | 26.38 | +3.16 | [-0.48, +6.90] |
| sft_v1_xifeval_sw | prompt_level_loose_acc,none | 17.25 | 18.18 | +0.93 | [-3.03, +4.90] |
| sft_v1_xifeval_sw | prompt_level_strict_acc,none | 14.92 | 18.18 | +3.26 | [-0.70, +7.23] |
| belebele_eng_Latn_locked | acc,none | 60.78 | 77.33 | +16.56 | [+13.33, +19.78] |
| belebele_eng_Latn_locked | acc_norm,none | 60.78 | 77.33 | +16.56 | [+13.33, +19.78] |
| belebele_swh_Latn_locked | acc,none | 28.33 | 29.89 | +1.56 | [-0.67, +3.67] |
| belebele_swh_Latn_locked | acc_norm,none | 28.33 | 29.89 | +1.56 | [-0.67, +3.67] |
| mmlu_locked | acc,none | 57.01 | 57.39 | +0.38 | [-0.17, +0.93] |
| global_mmlu_full_sw_locked | acc,none | 27.87 | 27.04 | -0.83 | [-1.87, +0.24] |

所有实际doc和prompt/options逐条相同。区间为描述性，未作多重校正，不用于挑任务或选模型。
未完成任务：
