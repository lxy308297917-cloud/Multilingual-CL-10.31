# 最新正式实验入口

本轮按照用户纠正后重新读取的ChatGPT计划执行。误转的旧跨层求解/SST任务已停止并保留产物。本轮正式比较仍在进行，不是已完成报告。用户随后授权的[FFT＋原36点表示保持补充实验](../fft_representation_20261109/README.md)是独立新增轨道，与已取消方案分开记录。

- [当前目标、固定算法和必要的基线对齐](current_objective.md)
- [真实训练进度](progress.md)
- [源模型与数据复用核对](reuse_audit.json)
- [实现定义和限制](implementation_clarifications.json)
- [代码与配置冻结记录](freeze.json)
- [预先固定的解释边界和历史强对照](limitations.md)
- [三语言完整范围结果表（缺失项留空）](results.md)

当前三语言主线新增训练计划14组，另复用SW batch2普通LoRA；加上独立FFT补充2组，共16组新增完整训练。每组6400条/400步，seed42；不因benchmark效果早停。SW旧36点FullRep保留为历史对照，新三语言均匀/随机/功能表示保护统一72点。额外3个对齐训练不是新增方法或参数搜索。

SW功能权重已用此前完整保存的build-only校准CE直接计算；AR/RU校准是同一固定算法所需的权重构造，不是小训练筛选。均匀、随机和功能权重的和均为6；权重上下界0.5至2。

完整正式测评已挂依赖队列，读取固定Final400；AR/RU官方知识任务已取得并锁定。每语言目标4项、通用6项。尚无新方法正式分数，不把CPU正确性或校准损失当作效果。后续形成结果表、配对区间、范数/漂移/成本报告，再精选上传GitHub。

## 环境与执行入口

实测环境见 [environments.json](environments.json)。训练／HF评分、vLLM生成与隔离EvalPlus执行使用三个独立环境；不能把其中一套依赖直接覆盖到另一套。editable源码与执行器补丁身份另见冻结记录，版本号本身不代替源码身份。

正式入口为 `scripts/function_aware_formal_20261109/run.py --language sw --method function`，语言可选sw/ar/ru，方法可选all/full/kd/random/function。先使用已有准备脚本在独立输出目录构建数据、缓存、固定权重与配置，再排队；不得在运行中的冻结目录重新执行构建脚本。工程检查只判运行正确性，随后从Base重新开始完整400步。

`collect_results.py`、`paired_results.py --partial`、`audit_artifacts.py` 和 `compute_costs.py` 为只读实验审计／报告工具，不启动训练或测评。缺失正式结果始终留空。公开代码中的 `/path/to/...` 是原服务器路径的替代标记，需要按本机已有资源调整。

严格提取数学与指令级结果另见 [补充指标表](auxiliary_results.md)。逐题数值矩阵带输入ID哈希，支持在不公开题面／模型答案的情况下复算配对变化；实际生成错误分析仍使用保留的本地原始记录。

## 正式测评之后的机制诊断

已准备并冻结 `diagnose_final.py` 与模型清单：仅在本轮全部113个正式测评阶段完成后，才允许申请GPU，对Base及已有21个适配权重作推理诊断。使用既有P1 check128（数学／指令各64条，曾用于旧诊断），与训练build128不重叠；不新增训练，不用于调系数或选checkpoint。

输出固定Final的36层／72点表示漂移及参考回答NLL，作为辅助解释，与下游成效表分开。当前状态为 **NOT_RUN，等待正式benchmark**。新增教师缓存与诊断耗时另计，不把NLL或低漂移当作方法收益。

离线复算入口为 `scripts/function_aware_formal_20261109/verify_numeric.py`，传入本目录并指定 `--family fa` 或 `--family fft`。当前部分单元格可复算通过；`--require-complete`会要求所有预定单元格齐全。验证范围是导出数值的一致性，原始输入与模型身份由独立服务器审计支持。

## 测评成本记录（CPU日志汇总）

[evaluation_costs.json](evaluation_costs.json)记录本轨道已完成测评的启动至完成墙钟、原始计时文件哈希、代码生成与实际隔离执行耗时；失败和复用单列。并行任务时长之和不是自然完成时间或CUDA内核时间，嵌套生成/执行时长不能重复相加。运行 `scripts/benchmark_5plus5_20261109/evaluation_costs.py` 只做CPU汇总，不启动实验；未完成和未知历史成本不填0。训练、教师缓存与校准成本仍见compute_costs.json。
