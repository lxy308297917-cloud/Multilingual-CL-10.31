# Qwen3 多语言持续学习

研究斯瓦希里语适配中的目标收益与通用能力变化。工作目录：`/data0/users/lixiaoyang/10.31/multilingual-research`。

## 当前任务

**寻找创新的my method，并完成与对应方法类型baseline的比较。** 已有seed42完整基线；剩余多种子复测已按用户要求停止。my method尚未实现、没有新方法分数。现有方法不能改名充当创新。

**交给ChatGPT时先读：[项目交接与研究问题](research/chatgpt_handoff.md)。**

## 已有结果

| 方法（6400条SW SFT，一轮400步） | SW阅读ACC | SW知识ACC | EN IFEval提示strict | GSM8K strict |
|---|---:|---:|---:|---:|
| 原模型 | 28.33 | 27.87 | 66.91 | 45.03 |
| Full FFT | 36.67 | 31.65 | 27.17 | 3.79 |
| LoRA r16 | 36.89 | 31.38 | 31.05 | 7.35 |
| SSU-Wanda | 35.44 | 31.24 | 47.50 | 13.80 |

数值为百分数。英语阅读/选择题知识提高，因此不是英语整体下降。数学格式、选择题偏置、SW ProX空输出、翻译重复分别诊断。完整任务及限制见[结果说明](results/README.md)；不能只看这张摘要表。

- [完整首轮结果 CSV](results/baselines_seed42.csv) · [JSON及模型身份](results/baselines_seed42.json)
- [可视结果表 XLSX](results/baselines.xlsx) · [已有141项完整任务快照](results/baselines_current.json)
- [旧MURI B独立结果](results/legacy_track_b.csv)：不同数据/预算/原HF协议，不能与新Aya结果混用。
- [当前状态](STATUS.md) · [逐项实验记录](research/sft_v1.md)

## 简洁目录

```text
configs/     模型/数据/上游版本、冻结预算、训练与评测配置
scripts/     数据准备、框架薄适配、训练/测评与结果导出
research/    协议、研究记录、诊断和ChatGPT交接
results/     可分享分数、CSV/XLSX、来源身份及诊断摘要
```

当前实现主要在`scripts/sft_v1/`和`configs/sft_v1/`。旧A线CPT、旧MURI SFT及4B准备记录保留追溯，不是当前GPU执行计划。冻结plan文件中的stage/next字段是当时的声明，不代表实时状态。

## 科学设置与入口

- Base：`Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，发布的后训练模型，不是另一个Qwen3-1.7B-Base。
- Aya SW6400条、一轮400步、有效batch16；2,884,423输入/1,078,347移位回答监督token；固定Final比较，不设英语保持准入门槛。
- FFT：LR1e-5、双卡ZeRO3；LoRA：r16/alpha32、LR1e-4；SSU-Wanda：锁定英语掩码、保护40.9594%唯一参数、LR1e-5/WD0；不是旧SSU15%。
- 成熟工具：LLaMA-Factory、lm-evaluation-harness、vLLM、BenchMAX与官方评分；版本见`configs/sft_v1/stack.lock.json`。
- 正式训练入口：`scripts/sft_v1/train_formal.py`、`train_ssu.py`；配置`configs/sft_v1/formal_sw_*_seed42.yaml`。
- CPUAdam接续用已验证`train_resume_safe.py`（SSU加`--ssu`）；canonical SSU resume整合尚未完成；全新训练不受该故障影响。
- 只启动手动队列；本账号最多4张GPU，检查进程归属、真实存储路径和余量，不操作其他用户。

代码依赖服务器外部的模型、数据缓存和环境，绝对路径需按实际机器调整；本仓库不是一键跨机器运行包。模型权重、数据集、逐题大日志和第三方源码不打包，版本/哈希及服务器来源保留在配置与结果中。

正式测试不用于调参、选种子或选择checkpoint。新方法首先明确假设、独立开发依据、同类型baseline与等预算口径，再冻结正式对照；NLL/PPL和机制代理不能替代主要benchmark结果。
