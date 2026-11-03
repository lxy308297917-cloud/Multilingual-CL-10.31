# 跨语言同题决策分析

每个语义ID在EN/SW/HA共同纳入，选项内容ID与金答案已对齐。阅读采用官方专业翻译；知识Full的翻译语义未经独立认证，分析仅探索性。不是不同语言非平行任务的直接作差。

编码顺序为EN/SW/HA：100表示仅EN正确，110表示EN和SW正确而HA错误，111三语言全对，000全错。选项排列和ID固定；学科和Base EN难度分层见机器结果。

| 模型 | 任务 | 读出 | N | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 | 三语言同内容决策% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | reading_exposed | 原字母 | 179 | 50 | 5 | 13 | 5 | 51 | 23 | 13 | 19 | 18.44 |
| Base | reading_exposed | 位置平均 | 179 | 26 | 5 | 15 | 2 | 66 | 12 | 37 | 16 | 12.29 |
| Base | reading_exposed | 内容均分 | 179 | 76 | 9 | 9 | 11 | 30 | 12 | 15 | 17 | 24.02 |
| Base | reading_exposed | 内容位置平均 | 179 | 69 | 13 | 13 | 9 | 29 | 9 | 16 | 21 | 27.93 |
| Base | knowledge_dev | 原字母 | 219 | 83 | 6 | 18 | 8 | 56 | 17 | 5 | 26 | 25.57 |
| Base | knowledge_dev | 位置平均 | 219 | 60 | 16 | 13 | 7 | 64 | 24 | 22 | 13 | 10.05 |
| Base | knowledge_dev | 内容均分 | 219 | 96 | 15 | 16 | 6 | 35 | 16 | 17 | 18 | 32.42 |
| Base | knowledge_dev | 内容位置平均 | 219 | 97 | 14 | 13 | 4 | 33 | 12 | 21 | 25 | 36.07 |
| SW FFT | reading_exposed | 原字母 | 179 | 33 | 6 | 8 | 3 | 50 | 25 | 29 | 25 | 18.99 |
| SW FFT | reading_exposed | 位置平均 | 179 | 23 | 9 | 8 | 2 | 58 | 19 | 36 | 24 | 15.08 |
| SW FFT | reading_exposed | 内容均分 | 179 | 46 | 6 | 8 | 2 | 52 | 16 | 21 | 28 | 27.93 |
| SW FFT | reading_exposed | 内容位置平均 | 179 | 45 | 3 | 6 | 4 | 54 | 14 | 26 | 27 | 25.14 |
| SW FFT | knowledge_dev | 原字母 | 219 | 77 | 8 | 7 | 11 | 60 | 15 | 16 | 25 | 21.46 |
| SW FFT | knowledge_dev | 位置平均 | 219 | 56 | 9 | 9 | 11 | 59 | 16 | 23 | 36 | 27.40 |
| SW FFT | knowledge_dev | 内容均分 | 219 | 86 | 8 | 6 | 4 | 47 | 18 | 20 | 30 | 31.96 |
| SW FFT | knowledge_dev | 内容位置平均 | 219 | 85 | 5 | 8 | 2 | 53 | 16 | 18 | 32 | 36.53 |
| SW→HA FFT | reading_exposed | 原字母 | 179 | 45 | 6 | 9 | 6 | 58 | 12 | 15 | 28 | 27.93 |
| SW→HA FFT | reading_exposed | 位置平均 | 179 | 23 | 8 | 8 | 5 | 56 | 20 | 30 | 29 | 17.88 |
| SW→HA FFT | reading_exposed | 内容均分 | 179 | 41 | 7 | 13 | 7 | 59 | 12 | 14 | 26 | 22.91 |
| SW→HA FFT | reading_exposed | 内容位置平均 | 179 | 42 | 7 | 11 | 5 | 56 | 15 | 16 | 27 | 24.02 |
| SW→HA FFT | knowledge_dev | 原字母 | 219 | 98 | 4 | 3 | 10 | 56 | 8 | 6 | 34 | 45.66 |
| SW→HA FFT | knowledge_dev | 位置平均 | 219 | 47 | 12 | 13 | 12 | 65 | 26 | 16 | 28 | 21.92 |
| SW→HA FFT | knowledge_dev | 内容均分 | 219 | 88 | 10 | 8 | 8 | 43 | 16 | 16 | 30 | 36.53 |
| SW→HA FFT | knowledge_dev | 内容位置平均 | 219 | 84 | 8 | 7 | 4 | 53 | 11 | 16 | 36 | 36.53 |
| HA-only FFT | reading_exposed | 原字母 | 179 | 50 | 1 | 3 | 5 | 75 | 5 | 0 | 40 | 46.37 |
| HA-only FFT | reading_exposed | 位置平均 | 179 | 22 | 5 | 11 | 3 | 63 | 20 | 24 | 31 | 20.11 |
| HA-only FFT | reading_exposed | 内容均分 | 179 | 37 | 6 | 9 | 5 | 62 | 15 | 20 | 25 | 21.23 |
| HA-only FFT | reading_exposed | 内容位置平均 | 179 | 39 | 6 | 7 | 3 | 65 | 15 | 15 | 29 | 22.35 |
| HA-only FFT | knowledge_dev | 原字母 | 219 | 103 | 1 | 1 | 8 | 55 | 3 | 3 | 45 | 58.90 |
| HA-only FFT | knowledge_dev | 位置平均 | 219 | 56 | 6 | 9 | 15 | 65 | 21 | 17 | 30 | 22.83 |
| HA-only FFT | knowledge_dev | 内容均分 | 219 | 76 | 12 | 7 | 8 | 52 | 18 | 14 | 32 | 30.59 |
| HA-only FFT | knowledge_dev | 内容位置平均 | 219 | 68 | 12 | 12 | 3 | 54 | 17 | 20 | 33 | 31.05 |
| SW SST NoRefresh | reading_exposed | 原字母 | 179 | 41 | 0 | 1 | 10 | 82 | 3 | 6 | 36 | 30.73 |
| SW SST NoRefresh | reading_exposed | 位置平均 | 179 | 26 | 7 | 6 | 4 | 59 | 21 | 37 | 19 | 13.97 |
| SW SST NoRefresh | reading_exposed | 内容均分 | 179 | 42 | 6 | 8 | 7 | 57 | 20 | 25 | 14 | 13.41 |
| SW SST NoRefresh | reading_exposed | 内容位置平均 | 179 | 36 | 6 | 6 | 4 | 57 | 18 | 34 | 18 | 15.08 |
| SW SST NoRefresh | knowledge_dev | 原字母 | 219 | 80 | 0 | 0 | 22 | 79 | 5 | 2 | 31 | 30.14 |
| SW SST NoRefresh | knowledge_dev | 位置平均 | 219 | 60 | 8 | 7 | 14 | 60 | 24 | 17 | 29 | 20.55 |
| SW SST NoRefresh | knowledge_dev | 内容均分 | 219 | 73 | 14 | 9 | 8 | 49 | 24 | 25 | 17 | 17.35 |
| SW SST NoRefresh | knowledge_dev | 内容位置平均 | 219 | 72 | 14 | 6 | 7 | 52 | 22 | 22 | 24 | 23.29 |
| SW SST R10 | reading_exposed | 原字母 | 179 | 41 | 0 | 1 | 10 | 82 | 4 | 6 | 35 | 30.17 |
| SW SST R10 | reading_exposed | 位置平均 | 179 | 27 | 7 | 6 | 4 | 59 | 20 | 37 | 19 | 13.97 |
| SW SST R10 | reading_exposed | 内容均分 | 179 | 42 | 6 | 8 | 7 | 57 | 20 | 25 | 14 | 13.41 |
| SW SST R10 | reading_exposed | 内容位置平均 | 179 | 36 | 6 | 6 | 4 | 57 | 18 | 34 | 18 | 14.53 |
| SW SST R10 | knowledge_dev | 原字母 | 219 | 80 | 0 | 0 | 22 | 79 | 5 | 2 | 31 | 30.14 |
| SW SST R10 | knowledge_dev | 位置平均 | 219 | 59 | 9 | 7 | 14 | 60 | 24 | 17 | 29 | 20.55 |
| SW SST R10 | knowledge_dev | 内容均分 | 219 | 73 | 14 | 9 | 8 | 47 | 26 | 25 | 17 | 18.26 |
| SW SST R10 | knowledge_dev | 内容位置平均 | 219 | 72 | 14 | 6 | 7 | 53 | 21 | 23 | 23 | 22.83 |
| SW→HA SST NoRefresh | reading_exposed | 原字母 | 179 | 28 | 6 | 10 | 4 | 68 | 16 | 16 | 31 | 22.35 |
| SW→HA SST NoRefresh | reading_exposed | 位置平均 | 179 | 25 | 7 | 8 | 3 | 60 | 16 | 29 | 31 | 21.23 |
| SW→HA SST NoRefresh | reading_exposed | 内容均分 | 179 | 30 | 10 | 10 | 7 | 58 | 21 | 24 | 19 | 14.53 |
| SW→HA SST NoRefresh | reading_exposed | 内容位置平均 | 179 | 28 | 8 | 9 | 3 | 59 | 21 | 32 | 19 | 13.97 |
| SW→HA SST NoRefresh | knowledge_dev | 原字母 | 219 | 87 | 7 | 6 | 7 | 58 | 19 | 9 | 26 | 32.42 |
| SW→HA SST NoRefresh | knowledge_dev | 位置平均 | 219 | 63 | 12 | 6 | 15 | 50 | 24 | 22 | 27 | 19.18 |
| SW→HA SST NoRefresh | knowledge_dev | 内容均分 | 219 | 66 | 20 | 11 | 6 | 45 | 21 | 23 | 27 | 21.92 |
| SW→HA SST NoRefresh | knowledge_dev | 内容位置平均 | 219 | 68 | 11 | 8 | 3 | 48 | 29 | 22 | 30 | 27.40 |

三语言选同一个内容既可能共同答对，也可能共同答错；一致率必须与111和000一起解释。去标签内容续写会改变输入要求，不能把与字母形式的差别全当成“语义能力”。位置平均后仍有退化，更不能仅凭原始A偏好解释全部能力变化。

逐题变对/变错、内容决策改变率、全部学科结果与难度组结果保留于机器分析；不据其中高分子组重新筛选方法或题目。
