# mtop：原生测试辅助指标

保留固定测试评分器已有指标；不替代主指标或作测试集调参。数值型比例在下表转换为百分数；ROUGE原本为0—100分，长度保留token。机器可读JSON保留原始单位。

| 模型 | 语言/任务 | 指标 | 值 |
|---|---|---|---:|
| base | es:mtop | hit_cap_rate | 4.07% |
| base | es:mtop | intent_accuracy | 12.51% |
| base | es:mtop | valid_tree_rate | 53.84% |
| base | th:mtop | hit_cap_rate | 2.17% |
| base | th:mtop | intent_accuracy | 0.11% |
| base | th:mtop | valid_tree_rate | 0.43% |
| base | fr:mtop | hit_cap_rate | 4.10% |
| base | fr:mtop | intent_accuracy | 0.00% |
| base | fr:mtop | valid_tree_rate | 0.00% |
| base | en:mtop | hit_cap_rate | 1.92% |
| base | en:mtop | intent_accuracy | 33.79% |
| base | en:mtop | valid_tree_rate | 88.58% |
| single_es | es:mtop | hit_cap_rate | 0.10% |
| single_es | es:mtop | intent_accuracy | 91.99% |
| single_es | es:mtop | valid_tree_rate | 99.90% |
| single_es | th:mtop | hit_cap_rate | 0.36% |
| single_es | th:mtop | intent_accuracy | 77.90% |
| single_es | th:mtop | valid_tree_rate | 98.88% |
| single_es | fr:mtop | hit_cap_rate | 0.50% |
| single_es | fr:mtop | intent_accuracy | 87.16% |
| single_es | fr:mtop | valid_tree_rate | 99.50% |
| single_es | en:mtop | hit_cap_rate | 0.16% |
| single_es | en:mtop | intent_accuracy | 87.85% |
| single_es | en:mtop | valid_tree_rate | 99.82% |
| single_th | es:mtop | hit_cap_rate | 2.00% |
| single_th | es:mtop | intent_accuracy | 88.19% |
| single_th | es:mtop | valid_tree_rate | 97.97% |
| single_th | th:mtop | hit_cap_rate | 0.29% |
| single_th | th:mtop | intent_accuracy | 87.67% |
| single_th | th:mtop | valid_tree_rate | 99.67% |
| single_th | fr:mtop | hit_cap_rate | 3.48% |
| single_th | fr:mtop | intent_accuracy | 83.71% |
| single_th | fr:mtop | valid_tree_rate | 96.46% |
| single_th | en:mtop | hit_cap_rate | 1.50% |
| single_th | en:mtop | intent_accuracy | 85.29% |
| single_th | en:mtop | valid_tree_rate | 98.45% |
| seq_es_th | es:mtop | hit_cap_rate | 1.27% |
| seq_es_th | es:mtop | intent_accuracy | 91.43% |
| seq_es_th | es:mtop | valid_tree_rate | 98.73% |
| seq_es_th | th:mtop | hit_cap_rate | 0.07% |
| seq_es_th | th:mtop | intent_accuracy | 89.19% |
| seq_es_th | th:mtop | valid_tree_rate | 99.93% |
| seq_es_th | fr:mtop | hit_cap_rate | 2.98% |
| seq_es_th | fr:mtop | intent_accuracy | 86.06% |
| seq_es_th | fr:mtop | valid_tree_rate | 96.87% |
| seq_es_th | en:mtop | hit_cap_rate | 0.93% |
| seq_es_th | en:mtop | intent_accuracy | 89.19% |
| seq_es_th | en:mtop | valid_tree_rate | 99.07% |
