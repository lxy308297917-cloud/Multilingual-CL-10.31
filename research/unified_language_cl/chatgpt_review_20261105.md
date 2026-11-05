# 交给 ChatGPT 的阶段性交接：2026-11-05

## 当前状态

七轨道、21段主训练、四语言原生dev/test矩阵、统一英语通用测评及全部预定中点已完成。六段Disjoint-ID补充训练目前一段完成，第二段正在运行，其余尚未执行；完整P3结论与总成本报告未完成。下列主实验结果均为已测实际成绩，未运行内容不冒充结果。

## 优先读取

1. [三角形阶段表：两个训练阶段、两个训练语言的实际成绩](triangle_tables.md)。
2. [所有主实验：ES/TH/FR/EN与统一通用任务](stage_language_matrix.md)。
3. [Single FFT及相对原模型的收益与区间](single_fft_results.md)。
4. [FFT-SEQ：TH阶段收益、ES保持、直接TH对照及区间](fft_seq_results.md)。
5. [196点固定开发学习曲线](development_curves.md)。
6. [数学负结果与评分/输出复核](math_output_diagnostics.md)、[EN阅读选项似然复核](../../results/unified_language_cl/en-reading-choice-audit-v1/report.md)。
7. [预定Same/Disjoint-ID资格](p3_eligibility.md)、[队列组成](../../results/unified_language_cl/p3-cohort-composition-v1/report.md)、[目前已完成的首个B直接TH模型](../../results/unified_language_cl/p3-progress-v1/completed_massive_direct_b.json)。
8. [实验来源与数据审计](dataset_audit.md)、[冻结评测协议](evaluation_protocol.md)、[代码整理和复现核验边界](reproduction.md)。

## 用户希望讨论的问题

用户观察到多数轨道旧任务遗忘较小。这与当前正式结果一致：多数ES同任务变化的条件配对95%区间跨零；主实验只有WikiLingua旧ES ROUGE-Lsum下降1.30分、区间[1.16,1.43]。不能把其他轨道的点估计下降写成已经确认的遗忘，也不能因为区间跨零就声称等价或永不遗忘。

另一个现象是多个轨道GSM8K/IFEval下降，但部分轨道EN阅读和同任务EN分数提升。因此“旧语言同任务保持”和“跨任务通用保持”应分开，不应把GSM8K、IFEval当作泰语bench，或合成为单一语言遗忘分数。

WikiLingua第二阶段TH ROUGE提升16.86分、ES下降1.30分；GSM8K灵活ACC下降20.77个百分点，IFEval提示级strict下降11.46个百分点。摘要长度、截断和语言输出变化仍可能影响这些成绩，不能据此直接宣称内部知识擦除。该轨道是官方划分的唯一映射非空子集，非完整官方test。

MathOctopus的Single ES、Single TH与SEQ在两个正式数学任务上均低于原模型。它目前不是一个“成功学习新语言再研究旧能力遗忘”的可靠正例，负结果保留。

## 讨论时需要保留的设置与限制

- 官方后训练Qwen3-1.7B，revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；Base是比较角色。
- 每语言阶段3200样本/200步；SIB为640样本/40步。AdamW LR1e-5、有效batch16、seed42、BF16、assistant-only、non-thinking、长度2048、无packing。
- Single TH与SEQ的TH实际输入/标签顺序一致；SEQ逻辑总预算长一阶段，ES前缀实际只训练一次并共享。
- 固定Final，不因英语下降早停，不用正式test选checkpoint、调学习率、追加预算或筛选种子。
- Same-ID主实验跨语言复用语义内容，可能影响保持；待完成的Disjoint-ID仍存在内容难度/标签组成差异，不能当纯语言切换因果。MASSIVE、XNLI、MTOP按预定资格全部纳入；不根据观察到的遗忘大小挑轨道。
- 条件配对区间不包含训练随机性且未校正多重比较；目前只做seed42，不额外训练种子。
- 三角表中尚未训练语言的空格只是展示省略，其实已测分数仍在完整矩阵；不同任务ACC、结构Tree EM和ROUGE不能直接平均。

请先据这些证据评估：当前任务/数据/预算设置为什么可能产生较少的旧语言同任务遗忘；哪些下降来自窄任务或输出读出适配；在P3完成前，哪些结论仍应保留。不要为了满足研究动机而把负结果美化或把遗忘作为必须做出来的指标。

## 交付范围

主实验全部聚合结果、区间、图表、科学代码和来源/协议锁已公开；本次新增三角表与上述交接说明、首个完成B模型的聚合分数。原始题目、参考摘要、完整树标签、逐样本预测与权重只在本地保留，不在仓库再分发。最终`same_vs_disjoint.md`、`cost_report.md`、`research_conclusion.md`须等所有预定对照完成后据实生成，目前不宣称全部任务结束。
