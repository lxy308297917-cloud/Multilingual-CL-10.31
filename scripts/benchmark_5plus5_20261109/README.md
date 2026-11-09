# 5+5补测入口

全部推理仍使用项目原有LM Evaluation Harness、vLLM和隔离EvalPlus执行器，没有新训练框架。

当前入口：

- `schedule_v3.py`：SIB-200及缺失历史RU Global-MMLU；读取deferred_models.json，跳过用户暂缓的未训练模型。
- `general_schedule.py`：只补历史LoRA缺失的3个HumanEval+、2个EN MMLU。
- `priority_evaluations_v2.py --family fa` 或 `--family fft`：保留原科学评测脚本，按最新优先级排已有模型的5+5任务，跳过额外翻译及用户暂缓模型。
- `evaluate_priority.py`、`gpu_slot_five.py`：仅资源适配，账号总量最多5，持有统一锁直到实际CUDA交接；不改变数据、权重、few-shot、batch、提示、解码或评分。
- `prepare.py`：固定SIB官方test、英文候选和字符长度归一化主指标，并记录与Belebele/训练集的文本包含检查。
- `report.py`：CPU逐题复核、标签分布/混淆矩阵、Base差值、配对区间及5+5结果表。`observe_v3.py`是有限期CPU报告刷新，不会启动训练。

`evaluate.py`保留已实际完成的历史AR Global-MMLU执行源码；`evaluate_v2.py`增加完整权重模型身份处理。早期schedule/observe版本及diagnostic_barrier仅为操作追溯，当前不运行；不要用旧调度器恢复暂停训练或机制诊断。运行记录、当前配置及临时5卡授权在research/benchmark_5plus5_20261109。

命令示例：`python scripts/benchmark_5plus5_20261109/report.py` 仅刷新结果，不启动GPU任务。实际手动启动记录另保存在本机日志；不要重复启动已有推理或覆盖输出。
