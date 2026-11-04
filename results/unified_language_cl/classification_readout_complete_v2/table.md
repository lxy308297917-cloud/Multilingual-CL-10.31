# 分类任务：固定开发64题读出复核

三个轨道、四个固定模型、四种语言均使用原先冻结的64题。生成与候选评分在同题上比较。候选使用全部合法标签的token log-prob之和，不做长度归一化。这里只诊断读出与标签格式，不替代正式全量测试ACC，不用于调参或选模型。

| 轨道 | 模型 | 语言 | n | 生成ACC% | 候选ACC% | 非法标签率% |
|---|---|---|---:|---:|---:|---:|
| massive | base | en | 64 | 35.94 | 37.50 | 7.81 |
| massive | base | es | 64 | 28.12 | 29.69 | 20.31 |
| massive | base | fr | 64 | 21.88 | 26.56 | 34.38 |
| massive | base | th | 64 | 1.56 | 12.50 | 78.12 |
| massive | single_es | en | 64 | 92.19 | 93.75 | 0.00 |
| massive | single_es | es | 64 | 90.62 | 92.19 | 0.00 |
| massive | single_es | fr | 64 | 87.50 | 87.50 | 0.00 |
| massive | single_es | th | 64 | 79.69 | 81.25 | 0.00 |
| massive | single_th | en | 64 | 89.06 | 89.06 | 0.00 |
| massive | single_th | es | 64 | 87.50 | 87.50 | 0.00 |
| massive | single_th | fr | 64 | 87.50 | 87.50 | 0.00 |
| massive | single_th | th | 64 | 84.38 | 84.38 | 0.00 |
| massive | seq_es_th | en | 64 | 95.31 | 95.31 | 0.00 |
| massive | seq_es_th | es | 64 | 90.62 | 89.06 | 0.00 |
| massive | seq_es_th | fr | 64 | 90.62 | 92.19 | 0.00 |
| massive | seq_es_th | th | 64 | 82.81 | 82.81 | 0.00 |
| xnli | base | en | 64 | 43.75 | 54.69 | 20.31 |
| xnli | base | es | 64 | 53.12 | 51.56 | 9.38 |
| xnli | base | fr | 64 | 60.94 | 68.75 | 6.25 |
| xnli | base | th | 64 | 51.56 | 48.44 | 1.56 |
| xnli | single_es | en | 64 | 84.38 | 84.38 | 0.00 |
| xnli | single_es | es | 64 | 84.38 | 84.38 | 0.00 |
| xnli | single_es | fr | 64 | 82.81 | 82.81 | 0.00 |
| xnli | single_es | th | 64 | 70.31 | 70.31 | 0.00 |
| xnli | single_th | en | 64 | 82.81 | 82.81 | 0.00 |
| xnli | single_th | es | 64 | 79.69 | 78.12 | 0.00 |
| xnli | single_th | fr | 64 | 81.25 | 81.25 | 0.00 |
| xnli | single_th | th | 64 | 76.56 | 78.12 | 0.00 |
| xnli | seq_es_th | en | 64 | 84.38 | 85.94 | 0.00 |
| xnli | seq_es_th | es | 64 | 84.38 | 84.38 | 0.00 |
| xnli | seq_es_th | fr | 64 | 82.81 | 84.38 | 0.00 |
| xnli | seq_es_th | th | 64 | 70.31 | 70.31 | 0.00 |
| sib200 | base | en | 64 | 42.19 | 42.19 | 0.00 |
| sib200 | base | es | 64 | 53.12 | 56.25 | 0.00 |
| sib200 | base | fr | 64 | 42.19 | 40.62 | 0.00 |
| sib200 | base | th | 64 | 14.06 | 48.44 | 78.12 |
| sib200 | single_es | en | 64 | 82.81 | 84.38 | 0.00 |
| sib200 | single_es | es | 64 | 82.81 | 85.94 | 0.00 |
| sib200 | single_es | fr | 64 | 79.69 | 79.69 | 0.00 |
| sib200 | single_es | th | 64 | 85.94 | 82.81 | 0.00 |
| sib200 | single_th | en | 64 | 82.81 | 82.81 | 0.00 |
| sib200 | single_th | es | 64 | 85.94 | 84.38 | 0.00 |
| sib200 | single_th | fr | 64 | 81.25 | 81.25 | 0.00 |
| sib200 | single_th | th | 64 | 84.38 | 84.38 | 0.00 |
| sib200 | seq_es_th | en | 64 | 84.38 | 84.38 | 0.00 |
| sib200 | seq_es_th | es | 64 | 87.50 | 87.50 | 0.00 |
| sib200 | seq_es_th | fr | 64 | 85.94 | 85.94 | 0.00 |
| sib200 | seq_es_th | th | 64 | 85.94 | 87.50 | 0.00 |

候选集合和任务提示已透露任务结构，候选得分也包含标签token偏好；其变化不能单独证明新增语言语义知识。64题的差异不等于全量正式测试差异。仅seed42，未新增GPU计算，未改变评分器或原有结果。
