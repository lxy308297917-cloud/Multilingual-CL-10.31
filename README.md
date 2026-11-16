# S3-Align：子空间对齐与英语保持

在 Qwen3-4B 的 Bactrian-X 斯瓦希里语指令微调中，研究跨语言表示对齐能否增加目标收益，以及英语表示保持能否减轻遗忘。

**训练使用 LLaMA-Factory，正式测评使用 lm-evaluation-harness。** 本仓库只保留方法扩展、运行入口和主要报告，不复制上游框架。

## 从哪里开始

| 需要做什么 | 入口 |
|---|---|
| 看原 S3 与英语保持效果（91 项分数） | [原始 SFT 报告](reports/sft/report.md) |
| 看共享子空间、检索与干预分析 | [机制报告](reports/mechanism/report.md) |
| 看语义、关键关系、数学遗忘及匹配训练 | [11 月 16 日结果快照](reports/mapping/report_draft.md) |
| 理解代码与复用上游框架 | [使用说明](docs/usage.md) |
| 找旧文件、历史实验和完整原版入口 | [迁移说明](docs/history.md) / [完整历史分支](https://github.com/lxy308297917-cloud/Multilingual-CL-10.31/tree/archive/research-through-20261116) |

目前证据支持英语保持能缓解部分英语能力损伤；原共享对齐没有可靠额外收益。新匹配训练快照中，T1 相对 T0 的 SW 阅读下降约 6.33 个百分点。该快照的 T2 尚未完成，不能将此状态当成实时实验进度，也不能提前判断 CCA 方法。具体协议、区间、单种子和材料暴露限制见报告。

## 现在的结构

```text
src/s3_align/  # 损失、CCA、训练扩展、预算记录和可选 vLLM 兼容
scripts/       # train.py / evaluate.py / analyze.py 三个入口
configs/       # 一份共享训练配方、一份资产路径模板
reports/       # sft / mechanism / mapping：原报告、图、数值表
docs/          # 使用及历史迁移说明
tests/         # 小型 CPU 回归检查
```

整理后的训练入口属于**开发版本**：数学核心与历史实现进行了 CPU 对照，尚未完成真实 GPU 训练回归。报告中的分数来自归档的冻结代码，不来自此次整理代码。训练需要原冻结语料、教师缓存、映射和标定文件；配置模板不能直接开跑。

```bash
# 无模型推理，复算已发布的 12 个汇总分数
python scripts/analyze.py \
  --vectors reports/mapping/analysis/formal_numeric.json \
  --scores reports/mapping/analysis/training_results.csv
```

本次仅整理 GitHub 展示与开发入口；服务器正在运行的实验和已有模型不迁移。历史 Git 对象仍保留，普通完整 clone 的体积不会因此立即变小；只需当前版本时可用 `git clone --depth 1`。
