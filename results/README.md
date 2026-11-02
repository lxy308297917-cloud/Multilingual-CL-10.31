# 已完成结果与诊断

## 阅读顺序

1. `baselines_seed42.csv` / `baselines_seed42.json`：完整首轮，Base、FFT、LoRA、SSU各20个官方指标，共80项。
3. `baselines_current.csv/json`：141个已完成指标。剩余复测已按用户要求停止；不是零分，也不是仍在队列中。
4. `legacy_track_b.csv/json/xlsx`：旧MURI实验独立原HF协议，5个固定身份、60项指标，不与新Aya/vLLM分数混合。
5. `diagnostics/`：官方分数对应的数学格式、IFEval类型、选择题偏置、ProX生成及翻译重复诊断。

## 首轮完整结果

ACC/EM/通过率为百分数，chrF++用原刻度。Base指发布的后训练Qwen3-1.7B，不是另一个Qwen3-1.7B-Base。

| 方法 | SW阅读ACC | SW知识ACC | EN→SW chrF++ | IFEval提示strict | GSM8K strict / flexible | SW→EN chrF++ |
|---|---:|---:|---:|---:|---:|---:|
| 原模型 | 28.33 | 27.87 | 5.11 | 66.91 | 45.03 / 73.01 | 22.91 |
| FFT | 36.67 | 31.65 | 8.10 | 27.17 | 3.79 / 61.26 | 14.53 |
| LoRA r16 | 36.89 | 31.38 | 7.20 | 31.05 | 7.35 / 56.79 | 16.20 |
| SSU-Wanda | 35.44 | 31.24 | 5.96 | 47.50 | 13.80 / 65.43 | 19.70 |

EN阅读ACC：Base60.78、FFT77.11、LoRA78.22、SSU79.67；EN MMLU：57.01、58.45、58.33、58.82。因此不能称为英语整体遗忘。SW指令指标几乎没有改善，全部保留在完整表中。

## 解释边界

- SW阅读/知识为选择题ACC；原模型选项偏置明显，收益可能包含校准变化。
- ProX生成EM受停止序列、空输出及提取影响，不能当作HF知识ACC。
- 数学strict是主指标，flexible为同生成的辅助提取；两者下降不能全归因于格式。
- EN→SW原模型439/500、FFT285/500出现至少10次连续重复的1–5词块；这是一项后验输出诊断，保留原chrF++。
- 三种子已有完整指标均逐项保留；样本标准差描述训练重复差异，不是测试抽样置信区间。未完成复测已停止，不承诺继续补齐。
- 旧MURI Full r2存在恢复实际LR偏差；r3已正确接续1250步并完成全部原HF测评，分别列示。

公开CSV/JSON保留80项完整首轮、141项全部已完成任务和60项旧MURI结果；来源以逻辑artifact名及SHA256表示。原始权重、数据、逐题大日志和服务器运行文件不包含。


## Benchmark v2

见[正式汇总](bench_v2/summary/paper_table.md)及[机器可读结果](bench_v2/summary/results.json)。代码生成完成不代表pass@1已运行；所有Code执行分数为null/NOT_RUN。
