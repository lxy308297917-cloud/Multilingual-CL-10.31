# 固定Final正式benchmark汇总

只汇入已验证完成任务；RUNNING/PENDING为尚无完整任务结果（包括排队、训练或测评中）。所有ACC为百分数，ROUGE/chrF保留原0–100尺度，不混合平均。

| Benchmark | Base | FFT | LoRA | SSU-Wanda | SST NoRefresh | SST R10 |
|---|---:|---:|---:|---:|---:|---:|
| SW Belebele ACC | 28.33 | 36.67 | 36.89 | 35.44 | 29.89 | 30.44 |
| SW Global-MMLU ACC | 27.87 | 31.65 | 31.38 | 31.24 | 27.04 | 27.09 |
| SW MGSM strict | 1.60 | 4.80 | 4.00 | 6.40 | 5.20 | 5.20 |
| SW MGSM flexible | 10.00 | 8.40 | 7.60 | 10.40 | 8.40 | 10.00 |
| SW XIFEval prompt strict | 14.92 | 14.69 | 14.92 | 14.45 | 18.18 | 18.41 |
| SW XL-Sum ROUGE-Lsum | 9.22 | 13.99 | 14.05 | 13.46 | 8.68 | 8.69 |
| EN Belebele ACC | 60.78 | 77.11 | 78.22 | 79.67 | 77.33 | 77.22 |
| EN MMLU ACC | 57.01 | 58.45 | 58.33 | 58.82 | 57.39 | 57.41 |
| EN IFEval prompt strict | 66.91 | 27.17 | 31.05 | 47.50 | 60.26 | 59.15 |
| EN GSM8K strict | 45.03 | 3.79 | 7.35 | 13.80 | 47.23 | 47.08 |
| EN GSM8K flexible | 73.01 | 61.26 | 56.79 | 65.43 | 69.14 | 68.84 |
| EN XL-Sum ROUGE-Lsum | 12.50 | 13.87 | 15.27 | 13.18 | 12.40 | 12.40 |
| EN→SW chrF++ full1012 | 5.40 | 8.00 | 7.13 | 6.13 | 5.88 | 5.94 |
| SW→EN chrF++ full1012 | 23.68 | 14.88 | 16.23 | 19.37 | 20.59 | 20.46 |

## Base→SW FFT→Hausa FFT

| Benchmark | Base | SW FFT Final | SW→HA FFT Final |
|---|---:|---:|---:|
| HA Belebele ACC | 23.89 | 28.44 | 26.89 |
| HA Global-MMLU ACC | 26.25 | 29.72 | 27.65 |
| SW Belebele ACC | 28.33 | 36.67 | 28.56 |
| SW Global-MMLU ACC | 27.87 | 31.65 | 27.47 |
| SW MGSM strict | 1.60 | 4.80 | 1.60 |
| SW MGSM flexible | 10.00 | 8.40 | 2.80 |
| SW XIFEval prompt strict | 14.92 | 14.69 | 13.99 |
| EN Belebele ACC | 60.78 | 77.11 | 69.11 |
| EN MMLU ACC | 57.01 | 58.45 | 56.89 |
| EN IFEval prompt strict | 66.91 | 27.17 | 25.88 |
| EN GSM8K strict | 45.03 | 3.79 | 0.15 |
| EN GSM8K flexible | 73.01 | 61.26 | 55.88 |

## Base→SW SST-NoRefresh→Hausa SST-NoRefresh

| Benchmark | Base | SW SST Final | SW→HA SST Final |
|---|---:|---:|---:|
| HA Belebele ACC | 23.89 | 28.67 | 26.89 |
| HA Global-MMLU ACC | 26.25 | 27.00 | 26.77 |
| SW Belebele ACC | 28.33 | 29.89 | 31.11 |
| SW Global-MMLU ACC | 27.87 | 27.04 | 28.69 |
| SW MGSM strict | 1.60 | 5.20 | 5.20 |
| SW MGSM flexible | 10.00 | 8.40 | 9.60 |
| SW XIFEval prompt strict | 14.92 | 18.18 | 15.62 |
| EN Belebele ACC | 60.78 | 77.33 | 77.67 |
| EN MMLU ACC | 57.01 | 57.39 | 56.53 |
| EN IFEval prompt strict | 66.91 | 60.26 | 56.01 |
| EN GSM8K strict | 45.03 | 47.23 | 21.08 |
| EN GSM8K flexible | 73.01 | 69.14 | 67.25 |

开发114题与有限FP32干预另表，不能与本表相减。SST两组full_determinism与旧FFT执行差异保留；所有方法均固定400步Final，无EN早停门槛。Hausa没有当前锁定版本的官方BenchMAX规则任务，未冒称已测。

代码benchmark四模型已发布，SST代码分数NOT_RUN，本轮固定SST主要清单不含该新增任务。
