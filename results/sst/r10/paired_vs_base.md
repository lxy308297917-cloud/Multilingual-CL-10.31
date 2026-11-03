# 固定阶段之间的配对正式变化

| Task | Metric | Before | After | Δ pp | 95% paired interval |
|---|---|---:|---:|---:|---:|
| belebele_eng_Latn_locked | acc,none | 60.78 | 77.22 | +16.44 | [+13.22, +19.67] |
| belebele_eng_Latn_locked | acc_norm,none | 60.78 | 77.22 | +16.44 | [+13.22, +19.67] |
| belebele_swh_Latn_locked | acc,none | 28.33 | 30.44 | +2.11 | [-0.11, +4.22] |
| belebele_swh_Latn_locked | acc_norm,none | 28.33 | 30.44 | +2.11 | [-0.11, +4.22] |
| sft_v1_gsm8k | exact_match,flexible-extract | 73.01 | 68.84 | -4.17 | [-6.60, -1.67] |
| sft_v1_gsm8k | exact_match,strict-match | 45.03 | 47.08 | +2.05 | [-0.91, +5.16] |
| sft_v1_ifeval | inst_level_loose_acc,none | 78.18 | 70.38 | -7.79 | [-10.60, -4.96] |
| sft_v1_ifeval | inst_level_strict_acc,none | 75.18 | 68.35 | -6.83 | [-9.63, -3.99] |
| sft_v1_ifeval | prompt_level_loose_acc,none | 70.43 | 61.18 | -9.24 | [-12.75, -5.55] |
| sft_v1_ifeval | prompt_level_strict_acc,none | 66.91 | 59.15 | -7.76 | [-11.46, -4.25] |
| sft_v1_mgsm_native_cot_sw | exact_match,flexible-extract | 10.00 | 10.00 | +0.00 | [-3.60, +3.60] |
| sft_v1_mgsm_native_cot_sw | exact_match,strict-match | 1.60 | 5.20 | +3.60 | [+1.20, +6.40] |
| sft_v1_xifeval_sw | inst_level_loose_acc,none | 25.91 | 26.38 | +0.47 | [-3.23, +4.30] |
| sft_v1_xifeval_sw | inst_level_strict_acc,none | 23.22 | 25.91 | +2.69 | [-0.94, +6.49] |
| sft_v1_xifeval_sw | prompt_level_loose_acc,none | 17.25 | 18.41 | +1.17 | [-2.80, +5.13] |
| sft_v1_xifeval_sw | prompt_level_strict_acc,none | 14.92 | 18.41 | +3.50 | [-0.47, +7.46] |
| global_mmlu_full_sw_locked | acc,none | 27.87 | 27.09 | -0.78 | [-1.82, +0.25] |
| mmlu_locked | acc,none | 57.01 | 57.41 | +0.41 | [-0.14, +0.98] |

实际doc、prompt与选项逐项相同。只比较完整固定Final的正式任务；无选模、无代理替换。区间未多重校正，保持IFEval提示内部指令cluster。
