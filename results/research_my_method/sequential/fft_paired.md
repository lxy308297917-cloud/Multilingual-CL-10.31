# 固定阶段之间的配对正式变化

| Task | Metric | Before | After | Δ pp | 95% paired interval |
|---|---|---:|---:|---:|---:|
| belebele_eng_Latn_locked | acc,none | 77.11 | 69.11 | -8.00 | [-10.11, -6.00] |
| belebele_eng_Latn_locked | acc_norm,none | 77.11 | 69.11 | -8.00 | [-10.11, -6.00] |
| belebele_swh_Latn_locked | acc,none | 36.67 | 28.56 | -8.11 | [-11.44, -4.78] |
| belebele_swh_Latn_locked | acc_norm,none | 36.67 | 28.56 | -8.11 | [-11.44, -4.78] |
| sft_v1_gsm8k | exact_match,flexible-extract | 61.26 | 55.88 | -5.38 | [-8.04, -2.88] |
| sft_v1_gsm8k | exact_match,strict-match | 3.79 | 0.15 | -3.64 | [-4.78, -2.65] |
| sft_v1_ifeval | inst_level_loose_acc,none | 41.49 | 40.17 | -1.32 | [-4.23, +1.60] |
| sft_v1_ifeval | inst_level_strict_acc,none | 41.01 | 39.93 | -1.08 | [-3.92, +1.82] |
| sft_v1_ifeval | prompt_level_loose_acc,none | 27.36 | 26.06 | -1.29 | [-4.62, +2.03] |
| sft_v1_ifeval | prompt_level_strict_acc,none | 27.17 | 25.88 | -1.29 | [-4.62, +2.03] |
| sft_v1_mgsm_native_cot_sw | exact_match,flexible-extract | 8.40 | 2.80 | -5.60 | [-9.20, -1.60] |
| sft_v1_mgsm_native_cot_sw | exact_match,strict-match | 4.80 | 1.60 | -3.20 | [-6.40, +0.00] |
| sft_v1_xifeval_sw | inst_level_loose_acc,none | 23.06 | 24.33 | +1.26 | [-2.04, +4.43] |
| sft_v1_xifeval_sw | inst_level_strict_acc,none | 23.06 | 24.33 | +1.26 | [-2.04, +4.43] |
| sft_v1_xifeval_sw | prompt_level_loose_acc,none | 14.69 | 13.99 | -0.70 | [-3.96, +2.56] |
| sft_v1_xifeval_sw | prompt_level_strict_acc,none | 14.69 | 13.99 | -0.70 | [-3.96, +2.56] |
| global_mmlu_full_sw_locked | acc,none | 31.65 | 27.47 | -4.18 | [-5.02, -3.36] |
| belebele_hau_Latn_sequential_locked | acc,none | 28.44 | 26.89 | -1.56 | [-5.11, +1.89] |
| belebele_hau_Latn_sequential_locked | acc_norm,none | 28.44 | 26.89 | -1.56 | [-5.11, +1.89] |
| global_mmlu_full_ha_locked | acc,none | 29.72 | 27.65 | -2.07 | [-2.89, -1.25] |
| mmlu_locked | acc,none | 58.45 | 56.89 | -1.56 | [-2.03, -1.08] |

实际doc、prompt与选项逐项相同。只比较完整固定Final的正式任务；无选模、无代理替换。区间未多重校正，保持IFEval提示内部指令cluster。
