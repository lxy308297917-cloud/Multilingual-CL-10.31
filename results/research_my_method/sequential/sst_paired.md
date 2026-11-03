# 固定阶段之间的配对正式变化

| Task | Metric | Before | After | Δ pp | 95% paired interval |
|---|---|---:|---:|---:|---:|
| belebele_eng_Latn_locked | acc,none | 77.33 | 77.67 | +0.33 | [-1.56, +2.22] |
| belebele_eng_Latn_locked | acc_norm,none | 77.33 | 77.67 | +0.33 | [-1.56, +2.22] |
| belebele_swh_Latn_locked | acc,none | 29.89 | 31.11 | +1.22 | [-2.33, +4.78] |
| belebele_swh_Latn_locked | acc_norm,none | 29.89 | 31.11 | +1.22 | [-2.33, +4.78] |
| sft_v1_gsm8k | exact_match,flexible-extract | 69.14 | 67.25 | -1.90 | [-4.25, +0.23] |
| sft_v1_gsm8k | exact_match,strict-match | 47.23 | 21.08 | -26.16 | [-28.96, -23.35] |
| sft_v1_ifeval | inst_level_loose_acc,none | 71.82 | 67.99 | -3.84 | [-6.49, -1.00] |
| sft_v1_ifeval | inst_level_strict_acc,none | 69.18 | 66.19 | -3.00 | [-5.56, -0.12] |
| sft_v1_ifeval | prompt_level_loose_acc,none | 62.48 | 57.49 | -4.99 | [-8.32, -1.48] |
| sft_v1_ifeval | prompt_level_strict_acc,none | 60.26 | 56.01 | -4.25 | [-7.58, -0.74] |
| sft_v1_mgsm_native_cot_sw | exact_match,flexible-extract | 8.40 | 9.60 | +1.20 | [-1.60, +4.00] |
| sft_v1_mgsm_native_cot_sw | exact_match,strict-match | 5.20 | 5.20 | +0.00 | [-2.40, +2.40] |
| sft_v1_xifeval_sw | inst_level_loose_acc,none | 26.86 | 25.12 | -1.74 | [-4.98, +1.39] |
| sft_v1_xifeval_sw | inst_level_strict_acc,none | 26.38 | 24.96 | -1.42 | [-4.57, +1.52] |
| sft_v1_xifeval_sw | prompt_level_loose_acc,none | 18.18 | 15.62 | -2.56 | [-5.36, +0.23] |
| sft_v1_xifeval_sw | prompt_level_strict_acc,none | 18.18 | 15.62 | -2.56 | [-5.36, +0.23] |
| global_mmlu_full_sw_locked | acc,none | 27.04 | 28.69 | +1.65 | [+0.80, +2.51] |
| belebele_hau_Latn_sequential_locked | acc,none | 28.67 | 26.89 | -1.78 | [-6.00, +2.56] |
| belebele_hau_Latn_sequential_locked | acc_norm,none | 28.67 | 26.89 | -1.78 | [-6.00, +2.56] |
| global_mmlu_full_ha_locked | acc,none | 27.00 | 26.77 | -0.23 | [-1.26, +0.81] |
| mmlu_locked | acc,none | 57.39 | 56.53 | -0.86 | [-1.32, -0.42] |

实际doc、prompt与选项逐项相同。只比较完整固定Final的正式任务；无选模、无代理替换。区间未多重校正，保持IFEval提示内部指令cluster。
