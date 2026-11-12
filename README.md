# Multilingual-CL：最新研究入口（2026-11-12）

## 完成：single 跨语言迁移矩阵与 XQuAD 补测

- [研究问题、完整矩阵、配对区间与研究定位](research/single_cross_language_20261111/README.md)
- [60项五任务分数](research/single_cross_language_20261111/scores.csv) · [90组配对区间](research/single_cross_language_20261111/paired_changes.csv)
- [生成式XQuAD报告](research/single_cross_language_20261111/xquad_report.md) · [16项分数](research/single_cross_language_20261111/xquad_scores.csv) · [英语对照](research/single_cross_language_20261111/english_anchor_scores.csv)

阅读收益可由其他语言训练大量获得；摘要仍有语言匹配优势，XNLI迁移具有方向性。旧SW→AR→RU设置的任务收益提前出现，不能作为干净的语言特异学习/遗忘基准。跨模型结果未因果隔离指令、格式、内容和语言学习；不据这些已暴露测试调出预期曲线。

另行开展Mistral-7B-v0.1英文Alpaca→11语言MultiAlpaca全参数复现，目前P1训练中，尚无P1/P2完整下游分数。Base测评与工程通过不等于两阶段方法有效。本次归档不宣称该复现已完成。

## 普通LoRA顺序训练已完成：SW → AR → RU

- [单语言与顺序训练关键结论、完整分数及配对区间](research/sequential_lora_20261110/README.md)
- [四阶段结果图](research/sequential_lora_20261110/stage_scores.png) · [80项主分](research/sequential_lora_20261110/scores.csv) · [120组配对区间](research/sequential_lora_20261110/paired_changes.csv)

单语言适配提升多项阅读、推理和摘要任务，但伴随指令、数学与代码代价。顺序训练存在提前跨语言收益和任务特异的遗忘；AR/RU本阶段并非所有任务上涨，未显示稳定优于独立适配。单顺序、单seed及评分接口敏感性限制一并报告，不据已暴露测试挑选有利协议。

下列内容为此前独立适配阶段；其中“不是顺序CL”描述其独立训练，不包括上述新增顺序实验。暂缓实验范围保持原记录。


## 上一阶段已完成：220/220与XQuAD10端点

已完成22个可运行模型/语言端点的全部5+5分数、AR/RU共10个XQuAD端点及逐题/配对核验。GPU已全部释放，没有接续训练或占卡。原计划的5个未启动训练、额外翻译和GPU机制诊断仍由用户暂缓；当前阶段完成不等于原研究整体全部完成。

- [完整正式报告：方法、三语言对齐表、强对照与限制](research/function_aware_formal_20261109/final_report.md)
- [ChatGPT交接与当前研究判断](research/benchmark_5plus5_20261109/chatgpt_handoff.md)
- [完整5+5主矩阵](research/benchmark_5plus5_20261109/results.md) · [阶段验收与暂缓范围](research/benchmark_5plus5_20261109/priority_completion.json)
- [历史micro1三语言基线](research/benchmark_5plus5_20261109/baseline_report.md)
- [XQuAD完整报告与格式诊断](research/xquad_20261109/final_report.md)
- [FFT已完成补测及匹配对照暂缓边界](research/fft_representation_20261109/README.md)

普通LoRA获得目标阅读/摘要等收益，并伴随部分数学、指令和代码退化。均匀和随机表示保持已是很强的baseline；当前静态功能风险加权尚无明确超越强对照的总体优势。三语言是独立适配，不是顺序CL或多训练种子证据。SIB归一化/标签偏好、同源文本重叠、XNLI选项偏好和XQuAD输出格式效应完整保留，不挑选有利指标宣布算法成功。

**历史指标更正保留：**5+5历史IFEval三行现为prompt strict：SW49.35%、AR42.70%、RU42.33%，不是instruction strict。原始预测未改，见[更正记录](research/benchmark_5plus5_20261109/metric_correction.json)。

公开脚本、配置、数值向量、身份哈希和报告支持CPU复算；原始题面、生成答案、权重和运行隐私信息留在服务器。墙钟成本、失败与复用分列，不将并行时长之和叫作自然耗时或GPU内核时间。

以下保留历史研究记录。历史“进行中/未运行”不表示恢复用户暂停任务。

## 最新：表示机制复现与 XNLI / HumanEval+ 补测

四模型固定端点的新增测评与表示分析已完成，无新训练。全表示保持相对普通LoRA的 SW XNLI、代码保持更好；跨语言平均余弦尚未提供独立额外解释。

- [本轮 ChatGPT 交接](research/representation_benchmarks_20261109/chatgpt_handoff.md)
- [九项主要任务与配对区间](research/representation_benchmarks_20261109/results.md)
- [完整机制报告和边界](research/representation_benchmarks_20261109/final_report.md)
- [科学代码与离线数值复算](scripts/representation_benchmarks/README.md)

## 此前 P2 训练与固定测评


SW三项P2训练与完整固定端点评测已完成：普通全层LoRA、训练中全表示保持、同池英语CE回放。40项绝对分数和48项配对比较均已复核。

- [最新ChatGPT交接](research/functional_representation_20261109/p2/chatgpt_handoff.md)
- [完整报告](research/functional_representation_20261109/p2/final_report.md) / [全部分数与区间](research/functional_representation_20261109/p2/results.md)
- [方法下一步决策](research/functional_representation_20261109/p2/next_method_decision.md)
- [公开科学代码和复算边界](scripts/functional_representation/p2/README.md)
- [此前三语言18组LoRA消融](research/lora_layer_analysis_20261108/final_report.md)
- [P0/P1功能敏感诊断](research/functional_representation_20261109/diagnosis.md)
- [前一轮4B泛化和翻译对照](research/updates_20261108/chatgpt_handoff.md)

普通全表示保持改善了本次SW适配的目标收益与通用保持权衡；同池回放出现数学负结果，完整保留。当前只有单SW、单seed42、单阶段证据，不是功能敏感/动态新方法或顺序CL有效性的证明。后续复杂版本、迁移和新翻译测评未运行。

下方保留历史研究记录，旧“进行中”状态不表示恢复已暂停任务。

---

<!-- aya4b-current-start -->
# Multilingual-CL：Qwen3-4B SW适配

## 本轮实验交付完成 · 2026-11-06

原模型、FFT、LoRA、SSU-Wanda 的固定预算正式测评已完成：**80 项分数、60 项配对比较**，32 个模型／任务组的权重与协议身份核验通过。逐题统计和翻译语料统计可在 CPU 上独立复算，不需原始题目或权重。

- [完整报告：设置、结果、成本与限制](research/aya_4b_20261105/final_report.md)
- [主结果与配对区间](research/aya_4b_20261105/results.md)
- [交给 ChatGPT 的总结](research/aya_4b_20261105/chatgpt_handoff.md)
- [机器可读结果](results/aya_4b_20261105/analysis-v1/scores.csv)
- [独立复算记录](results/aya_4b_20261105/analysis-v1/public_recomputation_verified.json)
- [复现说明](research/aya_4b_20261105/reproduction.md)

当前结果：阅读和摘要改善；SSU 的 SW 知识小幅提升；双向翻译仍落后于原模型。数学及指令遵循有下降，英语阅读和知识有提高，不能概括为英语整体退化。GSM8K strict/flexible 的差异另有提取敏感性统计。

## 翻译修正与补充诊断

两个预定重平衡数据 LoRA 候选均完成固定400步，实际训练曝光顺序一致。两组都未达到冻结开发条件，翻译修复尚未成功；有限搜索在两候选处结束。候选正式确认和保持测评NOT_RUN，不改原四模型结果，不用正式test调参。

- [最终研究结论和后续讨论](research/aya_4b_20261105/final_decision.md)
- [补充XNLI/XCOPA全量结果](research/aya_4b_20261105/supplemental_benchmarks.md)
- [第二个翻译候选](research/aya_4b_20261105/translation_lr2e5_candidate_result.md)
- [第一个候选：完整结果和边界](research/aya_4b_20261105/translation_rebalanced_candidate_result.md)
- [有限修正计划](research/aya_4b_20261105/translation_repair_candidate_plan.md)
- [XNLI/XCOPA 与开发诊断的阶段记录](research/aya_4b_20261105/progress.md)

主线加补充共92项分数、69项配对比较；开发诊断另列。

只有一个训练种子；样本区间不代表跨训练种子稳定性。模型规模和数据同时改变，不能分别归因。旧路线 B 仍保持用户停止。以下保留历史项目记录，历史“进行中”字样不表示任务恢复。
<!-- aya4b-current-end -->

---

## 历史研究记录

## 进行中的统一语言顺序基线 · 2026-11-04

## 当前研究（2026-11-05）

[三语言数学FFT完整路线A报告](research/math_sft_base_fft/final_report.md)：九语言Base普查、SW/BN/JA完整两轮FFT、六节点全部目标数学与通用benchmark、实际代码执行及配对分析已完成。路线A两个预算均未通过预定门槛，备用路线B正在准备数据、尚未训练，整体研究仍在执行。详见[给ChatGPT的交接](research/math_sft_base_fft/chatgpt_handoff.md)。


## 统一语言顺序实验：当前阶段

七轨道主训练、四语言原生与通用正式矩阵及全部预定中点评测已完成。MASSIVE/XNLI优先Disjoint-ID对照正在训练与测评，MTOP两段将在优先结果完成后衔接；整个目标继续进行。

- [七轨道研究入口](research/unified_language_cl/README.md)
- [Single FFT结果](research/unified_language_cl/single_fft_results.md)；[FFT-SEQ结果](research/unified_language_cl/fft_seq_results.md)
- [完整阶段与通用能力矩阵](research/unified_language_cl/stage_language_matrix.md)
- [196点固定开发学习曲线](research/unified_language_cl/development_curves.md)
- [43模块整理版代码与CPU核验范围](research/unified_language_cl/reproduction.md)

下面保留之前的研究记录；当前新增任务以统一语言实验入口为准，不自动开展My Method或第三语言训练。


MASSIVE和MathOctopus首批固定Final结果已完成：前者目标任务提升且GSM8K/IFEval退化，ES旧语言遗忘尚不明确；后者正式数学测试下降。其余五轨训练/测评和完整代码整理继续进行。

- [本轮进度、结果与限制](research/unified_language_cl/README.md)

---

## 最新研究交付 · 2026-11-04

固定100步决策稳定性pilot及全部评测已完成。内容KD较普通SFT保住部分分数，但没有优于同题回放；四组尚无明确Hausa净学习收益。

- [本轮研究入口](research/decision_stability/README.md)
- [完整结果与置信区间](research/decision_stability/results.md)
- [下一阶段决策](research/decision_stability/next_method_decision.md)

---

> 2026-11-03：新一轮[答案偏好与跨语言任务诊断](research/answer_bias_study/README.md)已完成；复用七个既有模型，无新增训练。原始正式benchmark不改。

## 2026-11-03 Hausa学习瓶颈诊断

[本轮结论与四项报告](research/next_stage/README.md) · [正式对照表](results/next_stage/paper_table.md) · [下一步研究决策](research/next_stage/next_method_decision.md)

下文是先前阶段的原始快照；本轮状态以上述入口为准。

# Multilingual-CL: SW adaptation and sequential learning

2026-11-03研究更新：SST NoRefresh/R10全套正式结果、SW→Hausa FFT、有限功能干预及文献排重已完成；尚无经验证的新方法。R10与NoRefresh第二阶段的实际状态以结果快照为准。

- [本轮完整交接](research/chatgpt_handoff.md)
- [研究决策](research/my_method_20261103/final_recommendation.md) · [创新性矩阵](research/my_method_20261103/novelty_matrix.md)
- [SST复现与分数](research/sst_reproduction/reproduction_report.md) · [顺序FFT结果](research/my_method_20261103/sequential_results.md)
- [结果与图表索引](results/research_my_method/README.md)

> 2026-11-03：[代码 benchmark 已完整执行](research/bench_v2/code_execution.md)，HumanEval下降，MBPP变化较小；不是所有代码任务均退化。My Method文献/数据探索与SST验收仍进行中，尚无新方法正式成效。

> 当前机制诊断、报告与代码结果交付已完成：三组模块恢复／注入、激活补丁与对照，尚无清晰功能分离证据。见[研究报告](research/mechanism/mechanism_report.md)与[早间汇报](research/mechanism/morning_brief.md)。正式baseline结果保持原样。

## Benchmark v2 补充测评（2026-11-02完成）

固定Base/FFT/LoRA/SSU Final，全量生成60,352条；16组翻译/摘要正式评分及8组代码生成/静态检查完成，28项测试通过。SW摘要和EN→SW有收益，SW→EN下降，EN摘要没有一致遗忘。代码8组完整隔离执行已完成：HumanEval+四模型为64.63/56.71/50.00/57.32%，MBPP+为51.59/51.06/53.17/54.76%；CrossSum仅可行性审计。原机制诊断负面结果保留。

- [Benchmark v2报告](research/bench_v2/benchmark_report.md)
- [正式表与配对区间](results/bench_v2/summary/paper_table.md)
- [复现与限制](research/bench_v2/reproduce.md)

# Qwen3 Multilingual Continual Learning

斯瓦希里语适配中的目标收益、通用任务退化和方法研究。

## 给ChatGPT的阅读入口

1. [项目交接：当前任务、参数、研究问题及限制](research/chatgpt_handoff.md)
2. [全部结果及解释边界](results/README.md)
3. [完整seed42结果CSV](results/baselines_seed42.csv) / [JSON](results/baselines_seed42.json)
4. [已完成141项结果](results/baselines_current.csv) / [旧MURI独立结果](results/legacy_track_b.csv)

## 当前主任务

当前按用户新计划开展内部机制诊断：资产核实、参数/激活漂移、局部模块恢复与注入、同输入激活补丁。见[机制诊断入口](research/mechanism/README.md)。主要实验与报告均已完成，当前尚无可靠的功能分离证据。暂不直接设计或训练最终My Method；旧候选四组已取消，剩余多种子复测保持停止。

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




