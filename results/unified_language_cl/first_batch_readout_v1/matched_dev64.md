# 固定开发材料上的读出诊断

本表不是正式benchmark成绩，不用于选择checkpoint。生成ACC与候选分数ACC都在相同64题/语言计算；候选用合法标签token log-prob之和，未做长度归一化。

| 轨道 | 模型 | 语言 | n | 生成ACC | 候选ACC | 生成非法标签率 |
|---|---|---|---:|---:|---:|---:|
| massive | base | es | 64 | 28.12 | 29.69 | 20.31 |
| massive | base | th | 64 | 1.56 | 12.50 | 78.12 |
| massive | base | fr | 64 | 21.88 | 26.56 | 34.38 |
| massive | base | en | 64 | 35.94 | 37.50 | 7.81 |
| massive | single_es | es | 64 | 90.62 | 92.19 | 0.00 |
| massive | single_es | th | 64 | 79.69 | 81.25 | 0.00 |
| massive | single_es | fr | 64 | 87.50 | 87.50 | 0.00 |
| massive | single_es | en | 64 | 92.19 | 93.75 | 0.00 |
| massive | single_th | es | 64 | 87.50 | 87.50 | 0.00 |
| massive | single_th | th | 64 | 84.38 | 84.38 | 0.00 |
| massive | single_th | fr | 64 | 87.50 | 87.50 | 0.00 |
| massive | single_th | en | 64 | 89.06 | 89.06 | 0.00 |
| massive | seq_es_th | es | 64 | 90.62 | 89.06 | 0.00 |
| massive | seq_es_th | th | 64 | 82.81 | 82.81 | 0.00 |
| massive | seq_es_th | fr | 64 | 90.62 | 92.19 | 0.00 |
| massive | seq_es_th | en | 64 | 95.31 | 95.31 | 0.00 |
| xnli | base | es | 64 | 53.12 | 51.56 | 9.38 |
| xnli | base | th | 64 | 51.56 | 48.44 | 1.56 |
| xnli | base | fr | 64 | 60.94 | 68.75 | 6.25 |
| xnli | base | en | 64 | 43.75 | 54.69 | 20.31 |
| sib200 | base | es | 64 | 53.12 | 56.25 | 0.00 |
| sib200 | base | th | 64 | 14.06 | 48.44 | 78.12 |
| sib200 | base | fr | 64 | 42.19 | 40.62 | 0.00 |
| sib200 | base | en | 64 | 42.19 | 42.19 | 0.00 |
