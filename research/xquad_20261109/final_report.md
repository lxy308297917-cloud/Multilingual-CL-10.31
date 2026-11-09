# XQuAD补测完整报告

2026-11-10T00:55:48.986956+08:00

## 完成范围

10/10个已存在固定端点完成，共11900条预测；20项绝对EM/F1及42项同语言配对区间通过逐题原始评分审计和独立数值复算。四个尚未训练的AR/RU模型遵从用户要求暂缓，不为补表重训；SW不受官方XQuAD支持。

## 全部结果

| 语言 | 模型 | EM% | F1% |
|---|---|---:|---:|
| ar | ar_Base | 0.08 | 26.46 |
| ar | ar_historical_lora | 49.33 | 66.71 |
| ru | ru_Base | 0.17 | 30.94 |
| ru | ru_historical_lora | 44.54 | 64.94 |
| ar | ar_all | 48.74 | 66.14 |
| ar | ar_function | 48.82 | 66.36 |
| ru | ru_all | 46.22 | 65.83 |
| ru | ru_full | 44.20 | 62.81 |
| ru | ru_random | 43.87 | 62.40 |
| ru | ru_function | 44.96 | 63.70 |

all为对齐普通LoRA；full为LoRA＋均匀72点表示保持，不是FFT；random为同权重多重集的随机分配；function为功能风险静态加权候选。historical_lora为历史micro1/GA16，新主线为micro2/GA8；各为固定Final400、有效batch16、seed42，不能混作训练种子复测。

## 候选方法的直接比较

| 语言 | 对照 | 指标 | 候选减对照（百分点） | 名义配对95%区间 |
|---|---|---|---:|---|
| ar | ar_all | EM | +0.08 | [-1.26, +1.44] |
| ar | ar_all | F1 | +0.21 | [-0.85, +1.26] |
| ru | ru_all | EM | -1.26 | [-2.64, +0.16] |
| ru | ru_all | F1 | -2.12 | [-3.25, -0.97] |
| ru | ru_full | EM | +0.76 | [-0.09, +1.60] |
| ru | ru_full | F1 | +0.89 | [+0.20, +1.66] |
| ru | ru_random | EM | +1.09 | [+0.24, +2.00] |
| ru | ru_random | F1 | +1.30 | [+0.42, +2.18] |

AR：候选相对普通LoRA的EM/F1区间均跨零，没有明确优势。RU：候选F1比普通LoRA低2.12个百分点，区间不跨零；EM差异区间跨零。候选RU F1高于均匀保护与随机分配，但不能据此宣布整体优于普通LoRA。所有区间均未作多重比较校正，是固定模型上的样本不确定性，不是跨训练种子稳定性或等效性检验。

## 冻结协议

- 模型为官方posttrained Qwen3-4B，revision 1cfa9a7208912126459214e8b04321603b3df60c，不是另一个Qwen3-4B-Base仓库。适配权重身份见configs/xquad_20261109/models.json及score_audit.json。
- 官方XQuAD revision 7d30520c717524000f0d9d2f9c10a069acd9d285；AR/RU各1190题、240个原始段落，保留原生Harness提示（包括原有阿拉伯语上下文标签）。
- 0-shot、non-thinking、HF BF16、greedy、最多128个新token、换行停止、batch4、seed42。原始模型输出直接评分，不增加答案提取或按测试选择提示。
- 官方SQuAD v1.1英文规范化、全部参考最大EM/F1。逐样本结果与框架聚合一致，完整保留模型输出。
- 配对bootstrap以240个原始段落为单位，同文问题一起保留，2000次重采样；不是1190题独立重采样。

## 输出格式限制

Base的严格EM很低，明显受到答案标签、附加词语、完整句与短答案参考的差异影响。历史LoRA与适配模型更倾向短答案。因此相对Base的大幅涨分包含格式适配，不能全部归因为语义理解提高。ASCII Markdown标点本身由官方规范化去除；真正保留的额外词语仍可能降低分数。format_diagnostics.json是后验辅助诊断，参考词序列包含率不是准确率，也可能奖励复制。未据该诊断更改主评分。

## 结论与后续边界

XQuAD补测已完成，但候选方法尚无跨语言、一致优于普通LoRA的问答证据。俄语F1负结果保留。固定5+5目标与通用能力尚未全部出齐，仍须完整比较学习—保持取舍；XQuAD不并入TargetACC4或GeneralAvg5。此次结果不用于修改系数、选checkpoint、增加种子或恢复用户暂缓的训练。

## 可复核材料

scores.csv、paired_changes.csv为完整表；score_audit.json对应本地原始11900条预测核验；numeric/*.json保留逐题ID、输入身份哈希、文章分组和数值分数，不公开题文。scripts/xquad_20261109/verify_numeric.py可离线复算20项分数和42项区间；其作用是数值一致性复算，官方评分正确性依赖另行保留的原始预测/参考审计。evaluation_costs.json记录运行成本，包含加载/CPU审计的墙钟时间不冒充CUDA kernel时间。
