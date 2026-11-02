# 代码 Benchmark 完整执行结果

同一冻结生成协议，greedy、每题1个样本。HumanEval164题，MBPP378题；Plus分数要求原始与增强测试均通过。

| 模型 | HumanEval pass@1 | HumanEval+ pass@1 | MBPP pass@1 | MBPP+ pass@1 |
|---|---:|---:|---:|---:|
| Base | 71.34% | 64.63% | 60.05% | 51.59% |
| FFT | 60.98% | 56.71% | 61.11% | 51.06% |
| LoRA | 54.88% | 50.00% | 62.96% | 53.17% |
| SSU-Wanda | 62.80% | 57.32% | 64.55% | 54.76% |

## 结论与边界

HumanEval两套测试均下降；MBPP原始测试三种适配均略升，增强测试FFT略降、LoRA与SSU略升。编程能力的变化依任务而异，不能以单项分数概括整体遗忘。逐题配对统计见 [paired-analysis.json](../../results/bench_v2/code_execution/paired-analysis.json)；置信区间是题目重采样，不能代替多训练种子。

## 评分与执行协议

- EvalPlus0.3.1，HumanEvalPlus v0.1.10，MbppPlus v0.2.0；fast_check=True，min_time_limit=1s，gt_time_limit_factor=4。
- 修复官方 HumanEval/32 find_zero 成功分支漏计 details/progress 的记账问题；断言与数值容差未改。四模型及标准答案一致使用修复版，原始安装包未改。
- 全部542条官方标准答案在相同隔离约束下通过，之后执行2168条冻结模型输出；没有重新生成或修补模型答案。
- 独立user/PID/network/mount namespace，非特权、只读运行时、无宿主数据/模型/GPU访问，seccomp与硬资源限制；每进程4GiB、最多2进程、2CPU、120s外层墙钟、80MiB有界tmpfs。并发2个隔离任务。
- 运行时锁定的是实际只读文件清单SHA256，不冒称容器镜像digest。
- Plus官方单独状态pass数不能直接作为Plus分子，必须与原始测试通过状态取交集。
- 官方fast_check状态不区分所有assertion、runtime error及内层超时；因此不捏造细分类错误率。外层无超时不表示所有单测均无超时。
- 先前896MiB版本HumanEval结果保留；统一4GiB复测逐题比较变化数：0。本表采用统一4GiB完整结果。

## 配对变化（百分点）

| 任务/指标 | 方法 | 相对Base变化 | 95%配对bootstrap区间 | 原来对→现在错 / 原来错→现在对 |
|---|---|---:|---:|---:|
| humaneval/base_pass | full | -10.37 | [-17.07, -3.66] | 25 / 8 |
| humaneval/plus_pass | full | -7.93 | [-14.63, -1.22] | 22 / 9 |
| humaneval/base_pass | lora | -16.46 | [-24.39, -9.15] | 36 / 9 |
| humaneval/plus_pass | lora | -14.63 | [-21.95, -7.32] | 33 / 9 |
| humaneval/base_pass | ssu | -8.54 | [-14.63, -2.44] | 21 / 7 |
| humaneval/plus_pass | ssu | -7.32 | [-13.41, -1.22] | 20 / 8 |
| mbpp/base_pass | full | +1.06 | [-3.44, +5.56] | 35 / 39 |
| mbpp/plus_pass | full | -0.53 | [-5.03, +3.97] | 37 / 35 |
| mbpp/base_pass | lora | +2.91 | [-1.59, +7.14] | 30 / 41 |
| mbpp/plus_pass | lora | +1.59 | [-2.65, +5.82] | 31 / 37 |
| mbpp/base_pass | ssu | +4.50 | [+0.53, +8.47] | 21 / 38 |
| mbpp/plus_pass | ssu | +3.17 | [-0.79, +7.14] | 23 / 35 |
