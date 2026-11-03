# SST正式训练与评测协议

## 冻结的科学设置

| 项目 | 共同设置 |
|---|---|
| 初始化 | 官方后训练Qwen/Qwen3-1.7B，revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| 数据 | 已冻结Aya Collection SW6400，同样本顺序、原LF tokenizer/loss mask |
| 预算 | 1 epoch，400真实optimizer步，effective batch16 |
| 实际暴露 | 6400条，2884423输入token，1078347移位assistant监督token |
| 模板 | qwen3 nonthinking，assistant-only，cutoff2048，无packing |
| 优化 | 原AdamW，LR1e−5，cosine，warmup0.05，weight_decay0，clip1 |
| 数值 | BF16前后向，FP32 Adam master/动量，SDPA，gradient checkpointing |
| 并行 | 两卡ZeRO3，microbatch1/卡，累积8；无optimizer offload |
| 随机性 | seed/data_seed42，full_determinism=True |
| 选择规则 | 固定400步Final；EN下降作为结果，不做baseline早停 |
| SST | 196矩阵、r=.99、精确gesvd；NoRefresh固定初始参考，R10每10真更新后刷新 |

原FFT默认非确定性执行；新SST启用确定性以满足原生恢复验收，差异明确保留。真实checkpoint100中FFT和NoRefresh底层Adam两组的LR均为8.94570254698197e−6，初始LR、betas(.9,.999)、eps1e−8和wd0一致；这验证保存点，不冒称逐步轨迹都做了独立LR检查。

NoRefresh两rank全部6400条实际暴露记录的输入/标签哈希、顺序、microstep与真实更新步逐项等于原FFT。仅写配置相同不足以替代这项核对。

## 初始基和恢复身份

实际正式两组引用已经验证的同一初始SVD cache，SHA256为`e76cf19c892984c8d9e5a648776a79e096a1db32a088b459162cc79d288a1918`。该文件约9.7GB，属于本地模型相关产物，不进入GitHub。每个目标矩阵FP32 master必须与cache参考逐值相同；rank和形状亦核对。

运行全确定性连续/恢复通过的是**该固定cache/实现身份下**的参数与基一致性。另一个环境从头重算SVD不能仅凭相同seed声称字节相同；SVD运行条件、CUDA库及精度都应另记。论文公式复现与字节级工程恢复是不同保证。

每100步保存模型、完整原生optimizer/scheduler/RNG、计数/样本暴露及SST参考/基/master哈希。已有失败目录与中间checkpoint保留，不覆盖、不冒充Final。数据与模型实际落盘data1，统一入口和日志记录在本地运行清单。

## 正式任务

| 能力 | 固定范围及指标 |
|---|---|
| SW/EN阅读 | Belebele各900题，0-shot聊天，4选项loglikelihood ACC，HF BF16 batch4 |
| SW知识 | Global-MMLU14042题/57科，每科5-shot，HF batch1，ACC |
| EN知识 | MMLU14042题/57科，5-shot，HF batch4，ACC |
| SW数学 | MGSM250题、native CoT 8-shot，strict/flexible最终答案ACC |
| SW指令 | BenchMAX XIFEval429提示/633指令，官方prompt/instruction strict/loose |
| EN数学 | GSM8K1319题、5-shot，strict与flexible同时报告 |
| EN指令 | IFEval541提示/834指令，官方strict/loose |
| 摘要 | v2 XL-Sum SW987、EN11535，冻结生成/ROUGE-Lsum等标准评分 |
| 翻译辅助 | FLORES v2 full1012，每方向语料级chrF++，保留旧500例成绩且分开 |

核心四项vLLM0.9.2，与旧模型同任务/模板/解码。摘要/翻译复用完成v2的实际输入哈希和生成参数。SW ROUGE仍沿锁定英语Punkt等既有协议限制；不事后更换以获得更好分数。

每任务完整成功后才计分；ACC独立从逐题结果复算并核对数量，配对比较核对实际doc、prompt和选项完全相同。选择题不会解析自由生成答案字母，因此没有将格式解析失败误当成选择题错误。生成任务同时报告严格/宽松标准及既有格式局限，不改原评分。

正式test不选rank、LR、刷新、语言、候选或checkpoint。有限114题FP32机制开发结果另表，不能与本表混作正式比较。
