# Qwen3 Multilingual Continual Learning

斯瓦希里语适配中的目标收益、通用任务退化和方法研究。

## 给ChatGPT的阅读入口

1. [项目交接：当前任务、参数、研究问题及限制](research/chatgpt_handoff.md)
2. [全部结果及解释边界](results/README.md)
3. [完整seed42结果CSV](results/baselines_seed42.csv) / [JSON](results/baselines_seed42.json)
4. [已完成141项结果](results/baselines_current.csv) / [旧MURI独立结果](results/legacy_track_b.csv)

## 当前主任务

寻找创新的my method，完成与对应方法类型baseline的比较。**my method尚未实现、没有方法分数。** 剩余多种子复测已按用户要求停止，已有完整结果保留。

| 方法 | SW阅读ACC | SW知识ACC | IFEval提示strict | GSM8K strict |
|---|---:|---:|---:|---:|
| 原模型 |28.33|27.87|66.91|45.03|
| FFT |36.67|31.65|27.17|3.79|
| LoRA |36.89|31.38|31.05|7.35|
| SSU-Wanda |35.44|31.24|47.50|13.80|

首轮：Qwen3-1.7B、Aya SW6400条、一轮400步Final。数值为百分数。英语阅读和选择题知识提高，不是英语整体下降；格式、选项偏置及翻译重复诊断另列，完整任务不能只用这张摘要表代替。

## 代码与配置

- `scripts/sft_v1/`：LLaMA-Factory薄适配、assistant-only曝光计数、SSU、CPUAdam接续、官方Harness观察与结果分析。
- `scripts/ssu_mask_adapter.py`：梯度掩码适配；其他`scripts/`文件是必要训练/评分辅助。
- `configs/sft_v1/formal_sw_*_seed42.yaml`：正式Full/LoRA/SSU设置；`stack.lock.json`记录上游版本。
- `configs/sft_v1/eval_tasks/`、`configs/eval_tasks/`：原生与HF任务定义。
- `results/`：官方分数CSV/JSON、诊断摘要；`publication_manifest.json`记录发布文件及原源代码哈希。

## 公开副本范围

当前上传精选的SFT代码、必要配置、结果及交接说明。服务器运行管理文件、完整内部资产清单、进程记录、AGENTS、权重、数据集和逐题大日志不发布。本地路径已替换为`/path/to/...`；服务器原始工程保持原样。代码不是开箱即用发行包：需要锁定上游环境、准备模型/数据/校准掩码及按机器设置路径；未宣称已在新机器重跑。

正式测试不用于调参或选种子。新方法先明确机制假设、独立开发依据、对应baseline及公平预算，再冻结正式比较；不把现有英语保护或SSU改名充当创新。
