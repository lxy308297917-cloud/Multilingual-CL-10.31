# Qwen3多语言适配内部机制诊断

当前任务：使用已有Base/FFT/LoRA/SSU固定400步Final，通过独立开发探针和局部干预检验语言接口与共享任务功能是否部分分离。完整结果与证据边界见[机制报告](mechanism_report.md)，科学运行核验摘要见[verification_summary](../../results/mechanism/verification_summary.json)。

## 阅读入口

- experiment_protocol.md：研究问题、执行顺序与证据边界。
- asset_inventory.md：模型、Adapter、各训练节点、模板和数据可用性。
- development_probes_v1.md：232个开发输入的来源、语言、评分和质量限制。
- activation_patching_v1.md：同输入补丁、随机与不同题对照及LL读出的局限。
- mechanism_report.md：Q1–Q4逐项证据，未完成不填结论。

## 实现与命令

scripts/mechanism包含参数/激活审计、有效权重和事务式模块干预、独立probe评分和推理、完整模块流水线、同输入参考LL补丁、条件效应矩阵和重复提示聚类分析。所有模型/数据/输出路径可由参数传入。输出目录必须是新目录，不改原始权重或baseline结果。

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests/mechanism -v
python scripts/mechanism/audit_parameter_drift.py --base /path/to/base --final /path/to/adapter --kind lora --output results/mechanism/new-parameter-run
python scripts/mechanism/module_swap.py --base /path/to/base --final /path/to/adapter --kind lora --probes /path/to/probes.jsonl --plan configs/mechanism/module_swap_v1.json --output results/mechanism/new-intervention-run
python scripts/mechanism/analyze_interventions.py --run results/mechanism/completed-run --probes /path/to/probes.jsonl --eos-id 151645 --parameter-drift /path/to/parameter_drift.json --output results/mechanism/new-analysis
```

默认device=cpu；真实GPU运行必须先检查共享资源与账号总量，并显式传入device及GPU UUID。现有torch/transformers/peft/numpy/scipy/pyarrow/matplotlib环境可用。测试使用随机小模型，不下载基础模型。三组真实流水线与同输入补丁另有独立运行证据，不能用toy测试代替。

## 解释

本轮不训练最终My Method，不扩种子，不用正式test搜索模块。参数/激活漂移仅作描述。模块替换是局部条件效应，完整Block还可能恢复LayerNorm，不能默认Attention/MLP差异就是纯交互。参考LL补丁可变化而实际任务准确率不变；需要结合已冻结任务干预及测量不确定性，不能强行宣称功能分离。
