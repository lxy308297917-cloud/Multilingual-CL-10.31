# Benchmark v2：Full正式结果

所有分数按0–100显示。NOT_RUN不等于0，运行中的任务不展示部分分数。

| 能力 | Benchmark | 方向 | 指标 | Base | FFT | LoRA | SSU |
|---|---|---|---|---:|---:|---:|---:|
| MT | flores200_en_sw_full_devtest_v2 | EN→SW | chrF++ | 5.40 | 8.00 | 7.13 | 6.13 |
| MT | flores200_en_sw_full_devtest_v2 | EN→SW | BLEU | 0.17 | 0.31 | 0.24 | 0.18 |
| MT | flores200_sw_en_full_devtest_v2 | SW→EN | chrF++ | 23.68 | 14.88 | 16.23 | 19.37 |
| MT | flores200_sw_en_full_devtest_v2 | SW→EN | BLEU | 4.36 | 2.22 | 2.66 | 2.50 |
| SUM | xlsum_sw_v2 | SW→SW | ROUGE-1 | 11.02 | 17.31 | 17.36 | 16.52 |
| SUM | xlsum_sw_v2 | SW→SW | ROUGE-2 | 2.88 | 4.55 | 4.48 | 4.40 |
| SUM | xlsum_sw_v2 | SW→SW | ROUGE-Lsum | 9.22 | 13.99 | 14.05 | 13.46 |
| SUM | xlsum_sw_v2 | SW→SW | chrF++ | 19.55 | 26.54 | 26.47 | 26.34 |
| SUM | xlsum_en_v2 | EN→EN | ROUGE-1 | 14.82 | 16.45 | 18.25 | 15.47 |
| SUM | xlsum_en_v2 | EN→EN | ROUGE-2 | 3.08 | 3.78 | 4.06 | 3.52 |
| SUM | xlsum_en_v2 | EN→EN | ROUGE-Lsum | 12.50 | 13.87 | 15.27 | 13.18 |
| SUM | xlsum_en_v2 | EN→EN | chrF++ | 21.05 | 22.02 | 23.48 | 21.51 |
| Code | humaneval_plus_v2 | EN→Python | pass@1_base | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Code | humaneval_plus_v2 | EN→Python | pass@1_plus | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Code | mbpp_plus_v2 | EN→Python | pass@1_base | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Code | mbpp_plus_v2 | EN→Python | pass@1_plus | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

## 微调相对Base变化（配对95%区间）

区间未作多重比较校正；只描述同一任务、同一语种、同一组ID的变化。

| Benchmark | 方法 | 指标 | Δ | 95%区间 |
|---|---|---|---:|---|
| flores200_en_sw_full_devtest_v2 | full | chrF++ | +2.60 | [+2.30, +2.90] |
| flores200_en_sw_full_devtest_v2 | full | BLEU | +0.13 | [+0.09, +0.18] |
| flores200_sw_en_full_devtest_v2 | full | chrF++ | -8.79 | [-9.50, -8.04] |
| flores200_sw_en_full_devtest_v2 | full | BLEU | -2.13 | [-2.79, -1.56] |
| xlsum_en_v2 | full | chrF++ | +0.96 | [+0.86, +1.06] |
| xlsum_en_v2 | full | ROUGE-1 | +1.63 | [+1.53, +1.72] |
| xlsum_en_v2 | full | ROUGE-2 | +0.70 | [+0.64, +0.76] |
| xlsum_en_v2 | full | ROUGE-Lsum | +1.37 | [+1.27, +1.45] |
| xlsum_sw_v2 | full | chrF++ | +6.99 | [+6.34, +7.58] |
| xlsum_sw_v2 | full | ROUGE-1 | +6.28 | [+5.69, +6.82] |
| xlsum_sw_v2 | full | ROUGE-2 | +1.67 | [+1.41, +1.93] |
| xlsum_sw_v2 | full | ROUGE-Lsum | +4.78 | [+4.29, +5.24] |
| flores200_en_sw_full_devtest_v2 | lora | chrF++ | +1.73 | [+1.44, +2.04] |
| flores200_en_sw_full_devtest_v2 | lora | BLEU | +0.07 | [+0.03, +0.11] |
| flores200_sw_en_full_devtest_v2 | lora | chrF++ | -7.45 | [-8.29, -6.57] |
| flores200_sw_en_full_devtest_v2 | lora | BLEU | -1.70 | [-2.35, -1.05] |
| xlsum_en_v2 | lora | chrF++ | +2.42 | [+2.33, +2.53] |
| xlsum_en_v2 | lora | ROUGE-1 | +3.43 | [+3.33, +3.54] |
| xlsum_en_v2 | lora | ROUGE-2 | +0.98 | [+0.92, +1.04] |
| xlsum_en_v2 | lora | ROUGE-Lsum | +2.77 | [+2.68, +2.86] |
| xlsum_sw_v2 | lora | chrF++ | +6.92 | [+6.26, +7.57] |
| xlsum_sw_v2 | lora | ROUGE-1 | +6.34 | [+5.78, +6.89] |
| xlsum_sw_v2 | lora | ROUGE-2 | +1.60 | [+1.29, +1.90] |
| xlsum_sw_v2 | lora | ROUGE-Lsum | +4.84 | [+4.36, +5.27] |
| flores200_en_sw_full_devtest_v2 | ssu | chrF++ | +0.73 | [+0.52, +0.96] |
| flores200_en_sw_full_devtest_v2 | ssu | BLEU | +0.01 | [-0.02, +0.04] |
| flores200_sw_en_full_devtest_v2 | ssu | chrF++ | -4.31 | [-5.26, -3.31] |
| flores200_sw_en_full_devtest_v2 | ssu | BLEU | -1.85 | [-2.46, -1.20] |
| xlsum_en_v2 | ssu | chrF++ | +0.46 | [+0.39, +0.53] |
| xlsum_en_v2 | ssu | ROUGE-1 | +0.65 | [+0.58, +0.73] |
| xlsum_en_v2 | ssu | ROUGE-2 | +0.44 | [+0.40, +0.49] |
| xlsum_en_v2 | ssu | ROUGE-Lsum | +0.68 | [+0.61, +0.75] |
| xlsum_sw_v2 | ssu | chrF++ | +6.79 | [+6.14, +7.38] |
| xlsum_sw_v2 | ssu | ROUGE-1 | +5.50 | [+4.96, +5.99] |
| xlsum_sw_v2 | ssu | ROUGE-2 | +1.52 | [+1.23, +1.77] |
| xlsum_sw_v2 | ssu | ROUGE-Lsum | +4.24 | [+3.79, +4.68] |
