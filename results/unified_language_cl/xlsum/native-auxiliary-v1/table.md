# xlsum：原生测试辅助指标

保留固定测试评分器已有指标；不替代主指标或作测试集调参。数值型比例在下表转换为百分数；ROUGE原本为0—100分，长度保留token。机器可读JSON保留原始单位。

| 模型 | 语言/任务 | 指标 | 值 |
|---|---|---|---:|
| base | en:xlsum | hit_cap_rate | 0.04% |
| base | en:xlsum | language_aux_confident_n | 11535 |
| base | en:xlsum | language_aux_correct_n | 11535 |
| base | en:xlsum | mean_generated_tokens | 166.98 tokens |
| base | en:xlsum | mean_reference_tokens | 28.28 tokens |
| base | en:xlsum | rouge1 | 15.67 points |
| base | en:xlsum | rouge2 | 3.38 points |
| base | en:xlsum | rougeLsum | 10.40 points |
| base | en:xlsum | source_truncated_rate | 1.35% |
| base | es:xlsum | hit_cap_rate | 3.28% |
| base | es:xlsum | language_aux_confident_n | 4762 |
| base | es:xlsum | language_aux_correct_n | 4761 |
| base | es:xlsum | mean_generated_tokens | 218.03 tokens |
| base | es:xlsum | mean_reference_tokens | 45.39 tokens |
| base | es:xlsum | rouge1 | 18.08 points |
| base | es:xlsum | rouge2 | 4.08 points |
| base | es:xlsum | rougeLsum | 11.88 points |
| base | es:xlsum | source_truncated_rate | 32.50% |
| base | fr:xlsum | hit_cap_rate | 0.00% |
| base | fr:xlsum | language_aux_confident_n | 1086 |
| base | fr:xlsum | language_aux_correct_n | 1086 |
| base | fr:xlsum | mean_generated_tokens | 136.01 tokens |
| base | fr:xlsum | mean_reference_tokens | 43.23 tokens |
| base | fr:xlsum | rouge1 | 23.48 points |
| base | fr:xlsum | rouge2 | 7.25 points |
| base | fr:xlsum | rougeLsum | 15.63 points |
| base | fr:xlsum | source_truncated_rate | 8.93% |
| base | th:xlsum | hit_cap_rate | 10.90% |
| base | th:xlsum | language_aux_confident_n | 826 |
| base | th:xlsum | language_aux_correct_n | 826 |
| base | th:xlsum | mean_generated_tokens | 346.32 tokens |
| base | th:xlsum | mean_reference_tokens | 126.59 tokens |
| base | th:xlsum | rouge1 | 27.59 points |
| base | th:xlsum | rouge2 | 9.12 points |
| base | th:xlsum | rougeLsum | 18.62 points |
| base | th:xlsum | source_truncated_rate | 59.08% |
| single_es | en:xlsum | hit_cap_rate | 0.18% |
| single_es | en:xlsum | language_aux_confident_n | 11527 |
| single_es | en:xlsum | language_aux_correct_n | 11527 |
| single_es | en:xlsum | mean_generated_tokens | 30.35 tokens |
| single_es | en:xlsum | mean_reference_tokens | 28.28 tokens |
| single_es | en:xlsum | rouge1 | 31.96 points |
| single_es | en:xlsum | rouge2 | 9.63 points |
| single_es | en:xlsum | rougeLsum | 23.81 points |
| single_es | en:xlsum | source_truncated_rate | 1.35% |
| single_es | es:xlsum | hit_cap_rate | 0.48% |
| single_es | es:xlsum | language_aux_confident_n | 4647 |
| single_es | es:xlsum | language_aux_correct_n | 4623 |
| single_es | es:xlsum | mean_generated_tokens | 37.99 tokens |
| single_es | es:xlsum | mean_reference_tokens | 45.39 tokens |
| single_es | es:xlsum | rouge1 | 28.53 points |
| single_es | es:xlsum | rouge2 | 8.80 points |
| single_es | es:xlsum | rougeLsum | 20.99 points |
| single_es | es:xlsum | source_truncated_rate | 32.50% |
| single_es | fr:xlsum | hit_cap_rate | 0.46% |
| single_es | fr:xlsum | language_aux_confident_n | 1082 |
| single_es | fr:xlsum | language_aux_correct_n | 1081 |
| single_es | fr:xlsum | mean_generated_tokens | 43.24 tokens |
| single_es | fr:xlsum | mean_reference_tokens | 43.23 tokens |
| single_es | fr:xlsum | rouge1 | 31.87 points |
| single_es | fr:xlsum | rouge2 | 11.83 points |
| single_es | fr:xlsum | rougeLsum | 24.14 points |
| single_es | fr:xlsum | source_truncated_rate | 8.93% |
| single_es | th:xlsum | hit_cap_rate | 60.90% |
| single_es | th:xlsum | language_aux_confident_n | 826 |
| single_es | th:xlsum | language_aux_correct_n | 826 |
| single_es | th:xlsum | mean_generated_tokens | 410.08 tokens |
| single_es | th:xlsum | mean_reference_tokens | 126.59 tokens |
| single_es | th:xlsum | rouge1 | 22.43 points |
| single_es | th:xlsum | rouge2 | 7.02 points |
| single_es | th:xlsum | rougeLsum | 16.53 points |
| single_es | th:xlsum | source_truncated_rate | 59.08% |
| single_th | en:xlsum | hit_cap_rate | 0.12% |
| single_th | en:xlsum | language_aux_confident_n | 11534 |
| single_th | en:xlsum | language_aux_correct_n | 11534 |
| single_th | en:xlsum | mean_generated_tokens | 48.95 tokens |
| single_th | en:xlsum | mean_reference_tokens | 28.28 tokens |
| single_th | en:xlsum | rouge1 | 28.89 points |
| single_th | en:xlsum | rouge2 | 8.25 points |
| single_th | en:xlsum | rougeLsum | 20.66 points |
| single_th | en:xlsum | source_truncated_rate | 1.35% |
| single_th | es:xlsum | hit_cap_rate | 0.59% |
| single_th | es:xlsum | language_aux_confident_n | 4760 |
| single_th | es:xlsum | language_aux_correct_n | 4754 |
| single_th | es:xlsum | mean_generated_tokens | 70.91 tokens |
| single_th | es:xlsum | mean_reference_tokens | 45.39 tokens |
| single_th | es:xlsum | rouge1 | 28.09 points |
| single_th | es:xlsum | rouge2 | 7.50 points |
| single_th | es:xlsum | rougeLsum | 19.44 points |
| single_th | es:xlsum | source_truncated_rate | 32.50% |
| single_th | fr:xlsum | hit_cap_rate | 0.28% |
| single_th | fr:xlsum | language_aux_confident_n | 1086 |
| single_th | fr:xlsum | language_aux_correct_n | 1086 |
| single_th | fr:xlsum | mean_generated_tokens | 63.26 tokens |
| single_th | fr:xlsum | mean_reference_tokens | 43.23 tokens |
| single_th | fr:xlsum | rouge1 | 31.52 points |
| single_th | fr:xlsum | rouge2 | 11.53 points |
| single_th | fr:xlsum | rougeLsum | 22.85 points |
| single_th | fr:xlsum | source_truncated_rate | 8.93% |
| single_th | th:xlsum | hit_cap_rate | 0.97% |
| single_th | th:xlsum | language_aux_confident_n | 826 |
| single_th | th:xlsum | language_aux_correct_n | 826 |
| single_th | th:xlsum | mean_generated_tokens | 140.79 tokens |
| single_th | th:xlsum | mean_reference_tokens | 126.59 tokens |
| single_th | th:xlsum | rouge1 | 35.24 points |
| single_th | th:xlsum | rouge2 | 13.02 points |
| single_th | th:xlsum | rougeLsum | 25.60 points |
| single_th | th:xlsum | source_truncated_rate | 59.08% |
| seq_es_th | en:xlsum | hit_cap_rate | 0.10% |
| seq_es_th | en:xlsum | language_aux_confident_n | 11533 |
| seq_es_th | en:xlsum | language_aux_correct_n | 11533 |
| seq_es_th | en:xlsum | mean_generated_tokens | 35.41 tokens |
| seq_es_th | en:xlsum | mean_reference_tokens | 28.28 tokens |
| seq_es_th | en:xlsum | rouge1 | 31.38 points |
| seq_es_th | en:xlsum | rouge2 | 9.42 points |
| seq_es_th | en:xlsum | rougeLsum | 23.03 points |
| seq_es_th | en:xlsum | source_truncated_rate | 1.35% |
| seq_es_th | es:xlsum | hit_cap_rate | 0.73% |
| seq_es_th | es:xlsum | language_aux_confident_n | 4692 |
| seq_es_th | es:xlsum | language_aux_correct_n | 4673 |
| seq_es_th | es:xlsum | mean_generated_tokens | 44.32 tokens |
| seq_es_th | es:xlsum | mean_reference_tokens | 45.39 tokens |
| seq_es_th | es:xlsum | rouge1 | 29.11 points |
| seq_es_th | es:xlsum | rouge2 | 8.89 points |
| seq_es_th | es:xlsum | rougeLsum | 21.15 points |
| seq_es_th | es:xlsum | source_truncated_rate | 32.50% |
| seq_es_th | fr:xlsum | hit_cap_rate | 0.55% |
| seq_es_th | fr:xlsum | language_aux_confident_n | 1083 |
| seq_es_th | fr:xlsum | language_aux_correct_n | 1082 |
| seq_es_th | fr:xlsum | mean_generated_tokens | 50.07 tokens |
| seq_es_th | fr:xlsum | mean_reference_tokens | 43.23 tokens |
| seq_es_th | fr:xlsum | rouge1 | 32.09 points |
| seq_es_th | fr:xlsum | rouge2 | 12.12 points |
| seq_es_th | fr:xlsum | rougeLsum | 24.04 points |
| seq_es_th | fr:xlsum | source_truncated_rate | 8.93% |
| seq_es_th | th:xlsum | hit_cap_rate | 0.85% |
| seq_es_th | th:xlsum | language_aux_confident_n | 826 |
| seq_es_th | th:xlsum | language_aux_correct_n | 826 |
| seq_es_th | th:xlsum | mean_generated_tokens | 137.81 tokens |
| seq_es_th | th:xlsum | mean_reference_tokens | 126.59 tokens |
| seq_es_th | th:xlsum | rouge1 | 35.50 points |
| seq_es_th | th:xlsum | rouge2 | 13.31 points |
| seq_es_th | th:xlsum | rougeLsum | 25.67 points |
| seq_es_th | th:xlsum | source_truncated_rate | 59.08% |
