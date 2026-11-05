# 训练及评分冻结协议 v1

冻结日期：2026-11-05；首轮新分数生成前冻结。

## 范围与选择

只研究官方 Qwen/Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` 的 Base 与独立单语言 FFT。九候选 SW/BN/TH/JA/ZH/RU/FR/DE/ES，英语仅作通用参照。

每语言 MGSM-Rev2 250、MSVAMP1000、PolyMath500（四难度各125）。MGSM/MSVAMP分别按 Base ACC 从低到高排名（同分平均名次），平均两名次；仅在平均名次完全相同的情况下优先脚本多样性，再按 ISO 字母序。训练数据至少1000条合格独立题的语言才可选；数据/解析异常先如实调查，不根据训练后的增益换语言。PolyMath不用于选语言。公布九语言全表。

这是 benchmark 引导的探索性语言、路线与预算选择，不是未暴露的独立确认实验。旧 ES/FR/TH 数学任务和英语测评已暴露。所有 E1/E2 完整报告，不隐藏低分。

## 提示与推理

MGSM/MSVAMP统一英文指令结构，指定在目标语言推理与回答：`Solve the following math problem in {language}. Show your reasoning. At the end, write #### followed by only the final numeric answer.` 后接两个换行与原题。统一取消旧提示中的字面占位词。因此九语言 Base 均重新运行；旧异提示结果不混用。0-shot。

PolyMath 使用官方 `instruction.py/query_dic` 的对应语言 boxed 提示，接在原题后两个换行；不增加 language_control forcing。LF 原版 qwen3 模板 enable_thinking=False。精确 token 输入在评测前锁定。

所有角色 BF16、TP1、greedy temperature0/top_p1/seed42，vLLM generation_config=vllm，max_model_len8192，max_num_seqs32/max_num_batched_tokens8192，GPU memory0.80。MGSM/MSVAMP生成上限1280，PolyMath2048；相同任务所有角色一致。

MGSM/MSVAMP主要指标最终数值ACC（冻结最后数值提取），另列 #### 严格提取ACC、解析率、触顶率和token长度。PolyMath严格复用官方第一个平衡 boxed 答案及 math_equal 表达式等价，不能替换成数值提取。四难度 low/medium/high/top 对应官方 low/middle/high/top，官方 DW-ACC 权重1/2/4/8除15；同时保留不舍入的计数和ACC，官方显示值按源码先逐难度round(%,1)，再加权round(%,1)。明确非thinking、2048预算，因此不宣称复现论文不同推理预算绝对分数。

## 路线A训练

各选定语言使用全部合格 MathOctopus 记录，不取共同交集，不任意抽5120或3200条。每语言从同一Base初始化，LR5e-6、AdamW、cosine、warmup0.05、weight_decay0、clip1.0、有效batch16、seed/data_seed42、BF16、assistant-only、non-thinking、长度2048、不packing。复用既有LLaMA-Factory+DeepSpeed ZeRO3。

持续完整两轮，同一个 optimizer 和 scheduler；保存真实epoch1/epoch2边界的完整权重，不能分别重启一轮或重置调度。不因通用能力下降早停。源记录不为凑16整除而丢弃；最后优化步可能小于16，分布式padding/重复曝光必须逐例记账。预期每轮ceil(N/16)，以实际dataloader与trace核实。

存储采用 save_strategy=epoch、save_only_model=true，避免每个节点重复写约20GB优化器状态。优化器在连续进程内保持不变；节点权重可独立加载，但不能冒充可恢复优化器状态。如故障无有效完整恢复状态，从同一Base重跑该角色到完整预算，在独立目录保留失败产物。训练保存后验证每轮源题覆盖、额外padding、token与监督mask，记录真实时间/GPU时。

## 完整角色测评

选定三语言 Base/E1/E2 各测 MGSM-Rev2、MSVAMP、PolyMath四难度与DW-ACC。各模型测 EN GSM8K(1319，原5-shot，flexible主/strict辅)、EN MMLU(原候选loglikelihood协议)、IFEval(541提示/834指令，官方strict提示级主)、HumanEval+(EvalPlus0.3.1，164题，固定greedy单样本实际pass@1，既有经验证namespace/seccomp沙箱)。本轮不添加 MBPP+ 或 EN Belebele。通用Base可共享，复用必须逐项证明身份一致。

报告 E1-Base/E2-Base/E2-E1，目标任务正确数/总数、right→wrong/wrong→right、配对bootstrap95%区间（10000次，seed42；问题ID成对抽样）。不把多种不同量纲强行平均。解析、截断与格式错误单独列。

## 路线门槛（训练前固定）

每语言一个固定epoch需 MGSM与MSVAMP各至少+2个百分点、至少一项配对95%区间下界>0，另一项不得明确退化。同一个epoch至少2/3语言通过，A即通过；E1或E2均可满足，但两个节点及第三语言结果全部保留。EN下降不构成失败。

仅当A的两个epoch都不达标，才启动路线B。仍选同三语言，mCoT-MATH每语言30000条合格、按源题去重/分层的记录，从Base独立两轮，完全同配方与全矩阵。若可用不足30000，运行前固定实际规则与数量；资源不足不得静默减预算。B之前核实数据、两盘余量与实际资源开销。B仍失败则停止扩展，不自动搜索LR/教师/更多epoch。

## 数据审计与公开范围

检查版本/哈希/非空、GSM8K原题身份及最终参考数、源题重复/语言/任务/长度、已知精确测试重叠。数字相同不等于同题；不宣称绝对无污染。抽样语义/推导核验明确样本量和局限，不把简单数值一致当整条推导正确。

不改旧结果；公开精选源码、可移植配置、聚合结果、协议与报告，不上传原题、答案、完整预测、权重、服务器私有队列/身份或内部资产清单。

## 后端身份补充（正式训练效果产生前记录）

本轮数字数学使用vLLM V1；既有EN GSM8K/IFEval使用V0，HumanEval+也在本轮所有角色统一固定V0。原11月2日HumanEval+生成用V1，不复用其旧Base分数。已核对新旧相同模型/输入token/题目/解码/源码/库版本/batch，但引擎日志分别确认V1与V0，164个净化程序中70个不同。因此本轮使用重新生成并实际执行的共享Base：HumanEval+108/164=65.85%，不能与旧V1 Base106/164混用。此差异不能解释为训练收益。所有当前E1/E2 code入口与新Base固定一致。
