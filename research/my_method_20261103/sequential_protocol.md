# Base→SW→Hausa 两阶段协议 v1

状态：数据与协议已冻结，FFT第二阶段400步完成；FFT固定阶段正式benchmark和各阶段HA参考完成；SST第二阶段训练及其正式评测接续。固定协议后才查看第二阶段正式测试，不能通过测试分数选语言、方法、rank或预算。

## 模型、数据与预算

- Stage0：官方后训练Qwen3-1.7B，revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e。
- Stage1：复用已有Aya Collection SW6400的一epoch/400步Final。FFT/LoRA/SSU已有；SST-NoRefresh400已完成，R10400运行且原生刷新恢复已验真。不得将早停或中间checkpoint充当Final。
- Stage2：Hausa6400，一epoch/400真optimizer步，有效batch16。原SW6400的11来源家族配额逐项相同；Hausa另留256开发例，不重复补足样本。
- SW训练2,884,423输入token、1,078,347 assistant监督token；HA训练2,596,327输入token、920,471 assistant监督token。**匹配的是样本数及来源配额，不是token或完全相同语义内容。** 所有第二阶段方法复用完全相同的HA样本、顺序和token，方法之间预算可比。
- 官方LLaMA-Factory qwen3 nonthinking、assistant-only、2048、无packing；seed/data_seed42，有效batch16，AdamW、LR1e-5、cosine、warmup0.05、weight_decay0、clip1、BF16、SDPA、gradient checkpointing。两卡时microbatch1、累积8；任何换卡数须保持有效batch及实际样本次序，不能静默改变。
- SST采用已验真的完整Adam更新投影，不投影原梯度替代。r99精确SVD；NoRefresh固定阶段参考；R10在每10个真更新后重算并重锚。
- SST工程发现非确定性BF16连续/恢复max差约3e-6，保留失败；full_determinism后NoRefresh全部17.2亿FP32值逐位相同。新正式SST与第二阶段统一启用确定性执行。旧阶段FFT默认非确定性，作为执行差异披露，不声称所有方法数值路径完全一致。

## 阶段边界

每阶段使用独立目录、完整日志、数据/配置/权重哈希。Stage2载入Stage1导出的Final模型参数，**重新初始化optimizer和400步scheduler**，step从0开始；不使用native resume把旧400步状态带到新数据，也不跳过HA前6400例。Stage2边界会把导出的BF16权重作为新起点，各方法一致。

主SST顺序版本以**当前SW模型W1**为第二阶段参考，重新SVD。W0全程锚定是另一个消融，不与主版本混合或复用W0初始基冒称W1基。共同W0基仅用于跨阶段几何分析。R10当前基与W0基分别记账。

## 最小对照与安排

1. FFT顺序适配：先取得普通完整预算的收益/遗忘现象。
2. SST-r99-NoRefresh顺序适配：阶段1通过固定Final后才接阶段2。
3. SST-r99-R10：实现与刷新恢复通过验收后训练；不为凑表提前写结果，不重复启动已有阶段。

先一个seed42，不新增多种子。优先完成已授权正式bench和既有SST。当前本账号最多4张GPU；所有串行衔接均先核验源任务退出码、Final完整性和GPU归属。

## 测量与数据使用

- Stage0、Stage1、Stage2分别报告HA新能力、SW已学能力、EN/通用旧能力。Belebele ACC和Global-MMLU ACC沿冻结官方任务形式；EN保持还须看GSM8K答案ACC、IFEval strict及已完成代码任务，不以EN知识单项代表全部能力。
- 原SW/EN正式范围沿现有协议；新HA若执行正式测评，先冻结原始数据版本、共同ID、900题Belebele/Global-MMLU全量范围、0/5-shot、chat模板及评分代码，再一次性运行。不能把新开发0-shot聊天MCQ分数与原5-shot全量正式分数混合。
- 开发：官方Global-MMLU dev285共同语义ID，57学科各3fit/2confirmation，共171/114。冻结时新方法尚未读取confirmation，现已完成一次预登记有限干预；EN/SW历史暴露、机器译文/低资源质量与无人工语义认证均明确标注。不能由这套探索开发分数宣称新方法正式有效。
- 语言指令任务缺失时记录缺失，不将自译Hausa规则冒称官方BenchMAX；目前锁定版本没有HA/YO/UR。
- 定义逐任务SW遗忘 F_SW=S_SW(W1)−S_SW(W2)，Hausa收益 G_HA=S_HA(W2)−S_HA(W1)，EN变化 S_EN(W2)−S_EN(W0)。保留各任务原始尺度和百分点，不平均ACC与chrF/ROUGE或用相对百分比混写。
- 比较新方法须同数据、同预算、同评测，至少匹配额外旧样本回放/梯度成本和实际有效权重变化范数。Final为主，不因EN下降停普通baseline。

## 交付边界

08:15不再启动长探索；08:30核验整合；09:00首版，10:00交接。若R10或第二阶段正式bench尚在运行，明确报告已完成范围、预估和剩余任务，不能将存活/排队写成完成。允许最终不推荐任何候选。

## 04:15实际排队

NoRefresh Stage1 Hausa两任务参考、完整Stage2训练及其正式old4/HA2任务已挂一次性依赖队列。仅在07:00之前、前置benchmark成功且两卡可用时启动400步，目标10:00前安全结束训练；晚于该时间窗留NOT_RUN。R10 Stage1尚未完成，因此Stage2 R10未启动，不将其填入完成表。

## 06:20实际执行状态

Base、SW FFT、SW NoRefresh的HA参考已齐；FFT阶段2全部固定任务已完成。NoRefresh阶段2满足前置后06:00实际启动，重算W1参考SVD并进行真实更新。正式old4/HA2评测只等待完整400 Final；R10阶段2仍NOT_RUN。
