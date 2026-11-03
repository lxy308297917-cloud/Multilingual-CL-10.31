# Base→Hausa-only固定Final400：完整预定正式指标

所有数值为百分数，变化为百分点。阅读含已曝光180开发题；全900及独立于本轮曲线的720拆分见四模型对照。区间为描述性、未多重校正；下表所有阅读行沿用原逐题配对区间；正文HA主要结论采用四模型对照的段落cluster区间，后者优先，数值会有小差异。

| Task | Metric | n | Base | HA-only | Δ pp | 95% paired interval |
|---|---|---:|---:|---:|---:|---:|
| belebele_eng_Latn_locked | acc,none | 900 | 60.78 | 70.33 | +9.56 | [+6.56,+12.56] |
| belebele_eng_Latn_locked | acc_norm,none | 900 | 60.78 | 70.33 | +9.56 | [+6.56,+12.56] |
| belebele_swh_Latn_locked | acc,none | 900 | 28.33 | 23.78 | -4.56 | [-8.89,-0.44] |
| belebele_swh_Latn_locked | acc_norm,none | 900 | 28.33 | 23.78 | -4.56 | [-8.89,-0.44] |
| sft_v1_gsm8k | exact_match,flexible-extract | 1319 | 73.01 | 59.14 | -13.87 | [-16.53,-11.22] |
| sft_v1_gsm8k | exact_match,strict-match | 1319 | 45.03 | 1.90 | -43.14 | [-45.87,-40.48] |
| sft_v1_ifeval | inst_level_loose_acc,none | 834 | 78.18 | 48.80 | -29.38 | [-33.09,-25.33] |
| sft_v1_ifeval | inst_level_strict_acc,none | 834 | 75.18 | 48.56 | -26.62 | [-30.43,-22.68] |
| sft_v1_ifeval | prompt_level_loose_acc,none | 541 | 70.43 | 34.01 | -36.41 | [-41.04,-31.42] |
| sft_v1_ifeval | prompt_level_strict_acc,none | 541 | 66.91 | 34.01 | -32.90 | [-37.89,-28.10] |
| sft_v1_mgsm_native_cot_sw | exact_match,flexible-extract | 250 | 10.00 | 2.40 | -7.60 | [-11.60,-3.60] |
| sft_v1_mgsm_native_cot_sw | exact_match,strict-match | 250 | 1.60 | 2.00 | +0.40 | [-1.60,+2.40] |
| sft_v1_xifeval_sw | inst_level_loose_acc,none | 633 | 25.91 | 27.49 | +1.58 | [-2.08,+5.28] |
| sft_v1_xifeval_sw | inst_level_strict_acc,none | 633 | 23.22 | 27.33 | +4.11 | [+0.62,+7.78] |
| sft_v1_xifeval_sw | prompt_level_loose_acc,none | 429 | 17.25 | 18.18 | +0.93 | [-2.80,+4.90] |
| sft_v1_xifeval_sw | prompt_level_strict_acc,none | 429 | 14.92 | 18.18 | +3.26 | [-0.23,+7.23] |
| global_mmlu_full_sw_locked | acc,none | 14042 | 27.87 | 26.63 | -1.24 | [-2.08,-0.39] |
| belebele_hau_Latn_sequential_locked | acc,none | 900 | 23.89 | 24.44 | +0.56 | [-2.22,+3.22] |
| belebele_hau_Latn_sequential_locked | acc_norm,none | 900 | 23.89 | 24.44 | +0.56 | [-2.22,+3.22] |
| global_mmlu_full_ha_locked | acc,none | 14042 | 26.25 | 27.52 | +1.27 | [+0.54,+2.01] |
| mmlu_locked | acc,none | 14042 | 57.01 | 56.62 | -0.38 | [-1.05,+0.26] |
