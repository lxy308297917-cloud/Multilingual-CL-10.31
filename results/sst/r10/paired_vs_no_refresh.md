# 固定阶段之间的配对正式变化

| Task | Metric | Before | After | Δ pp | 95% paired interval |
|---|---|---:|---:|---:|---:|
| belebele_eng_Latn_locked | acc,none | 77.33 | 77.22 | -0.11 | [-0.67, +0.44] |
| belebele_eng_Latn_locked | acc_norm,none | 77.33 | 77.22 | -0.11 | [-0.67, +0.44] |
| belebele_swh_Latn_locked | acc,none | 29.89 | 30.44 | +0.56 | [+0.00, +1.11] |
| belebele_swh_Latn_locked | acc_norm,none | 29.89 | 30.44 | +0.56 | [+0.00, +1.11] |
| sft_v1_gsm8k | exact_match,flexible-extract | 69.14 | 68.84 | -0.30 | [-1.44, +0.91] |
| sft_v1_gsm8k | exact_match,strict-match | 47.23 | 47.08 | -0.15 | [-1.52, +1.29] |
| sft_v1_ifeval | inst_level_loose_acc,none | 71.82 | 70.38 | -1.44 | [-3.18, +0.36] |
| sft_v1_ifeval | inst_level_strict_acc,none | 69.18 | 68.35 | -0.84 | [-2.65, +0.99] |
| sft_v1_ifeval | prompt_level_loose_acc,none | 62.48 | 61.18 | -1.29 | [-3.70, +1.11] |
| sft_v1_ifeval | prompt_level_strict_acc,none | 60.26 | 59.15 | -1.11 | [-3.51, +1.29] |
| sft_v1_mgsm_native_cot_sw | exact_match,flexible-extract | 8.40 | 10.00 | +1.60 | [+0.40, +3.20] |
| sft_v1_mgsm_native_cot_sw | exact_match,strict-match | 5.20 | 5.20 | +0.00 | [-1.20, +1.20] |
| sft_v1_xifeval_sw | inst_level_loose_acc,none | 26.86 | 26.38 | -0.47 | [-1.72, +0.66] |
| sft_v1_xifeval_sw | inst_level_strict_acc,none | 26.38 | 25.91 | -0.47 | [-1.58, +0.64] |
| sft_v1_xifeval_sw | prompt_level_loose_acc,none | 18.18 | 18.41 | +0.23 | [-0.70, +1.17] |
| sft_v1_xifeval_sw | prompt_level_strict_acc,none | 18.18 | 18.41 | +0.23 | [-0.70, +1.17] |
| global_mmlu_full_sw_locked | acc,none | 27.04 | 27.09 | +0.05 | [-0.08, +0.17] |
| mmlu_locked | acc,none | 57.39 | 57.41 | +0.02 | [-0.16, +0.21] |

实际doc、prompt与选项逐项相同。只比较完整固定Final的正式任务；无选模、无代理替换。区间未多重校正，保持IFEval提示内部指令cluster。
