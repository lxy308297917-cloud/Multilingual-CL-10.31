# 固定阶段之间的配对正式变化

| Task | Metric | Before | After | Δ pp | 95% paired interval |
|---|---|---:|---:|---:|---:|
| belebele_eng_Latn_locked | acc,none | 69.11 | 70.33 | +1.22 | [-0.89, +3.22] |
| belebele_eng_Latn_locked | acc_norm,none | 69.11 | 70.33 | +1.22 | [-0.89, +3.22] |
| belebele_swh_Latn_locked | acc,none | 28.56 | 23.78 | -4.78 | [-7.22, -2.33] |
| belebele_swh_Latn_locked | acc_norm,none | 28.56 | 23.78 | -4.78 | [-7.22, -2.33] |
| sft_v1_gsm8k | exact_match,flexible-extract | 55.88 | 59.14 | +3.26 | [+0.91, +5.61] |
| sft_v1_gsm8k | exact_match,strict-match | 0.15 | 1.90 | +1.74 | [+1.06, +2.50] |
| sft_v1_ifeval | inst_level_loose_acc,none | 40.17 | 48.80 | +8.63 | [+5.60, +11.65] |
| sft_v1_ifeval | inst_level_strict_acc,none | 39.93 | 48.56 | +8.63 | [+5.56, +11.67] |
| sft_v1_ifeval | prompt_level_loose_acc,none | 26.06 | 34.01 | +7.95 | [+4.07, +11.65] |
| sft_v1_ifeval | prompt_level_strict_acc,none | 25.88 | 34.01 | +8.13 | [+4.25, +11.83] |
| sft_v1_mgsm_native_cot_sw | exact_match,flexible-extract | 2.80 | 2.40 | -0.40 | [-3.20, +2.40] |
| sft_v1_mgsm_native_cot_sw | exact_match,strict-match | 1.60 | 2.00 | +0.40 | [-2.00, +2.80] |
| sft_v1_xifeval_sw | inst_level_loose_acc,none | 24.33 | 27.49 | +3.16 | [+0.47, +5.97] |
| sft_v1_xifeval_sw | inst_level_strict_acc,none | 24.33 | 27.33 | +3.00 | [+0.31, +5.86] |
| sft_v1_xifeval_sw | prompt_level_loose_acc,none | 13.99 | 18.18 | +4.20 | [+1.63, +6.99] |
| sft_v1_xifeval_sw | prompt_level_strict_acc,none | 13.99 | 18.18 | +4.20 | [+1.63, +6.99] |
| global_mmlu_full_sw_locked | acc,none | 27.47 | 26.63 | -0.84 | [-1.35, -0.31] |
| belebele_hau_Latn_sequential_locked | acc,none | 26.89 | 24.44 | -2.44 | [-4.67, -0.11] |
| belebele_hau_Latn_sequential_locked | acc_norm,none | 26.89 | 24.44 | -2.44 | [-4.67, -0.11] |
| global_mmlu_full_ha_locked | acc,none | 27.65 | 27.52 | -0.13 | [-0.63, +0.38] |
| mmlu_locked | acc,none | 56.89 | 56.62 | -0.26 | [-0.80, +0.24] |

实际doc、prompt与选项逐项相同。只比较完整固定Final的正式任务；无选模、无代理替换。区间未多重校正，保持IFEval提示内部指令cluster。
