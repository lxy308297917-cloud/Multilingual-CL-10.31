<!-- aya4b-current-start -->
# Multilingual-CL：Qwen3-4B SW适配

## 当前阶段交付 · 2026-11-06（仍在运行）

原Base/FFT/LoRA/SSU训练均已核验；完整正式测评尚未测齐。当前已核验的64项分数、44个配对比较及待完成项见[阶段结果与交接](research/aya_4b_20261105/progress.md)。翻译数据重平衡LoRA候选仍训练中，未把候选或未完成结果写入正式模型列。

- [本轮研究入口](research/aya_4b_20261105/README.md) · [复现代码与数据身份](research/aya_4b_20261105/reproduction.md)
- [新增XNLI/XCOPA完整比较](research/aya_4b_20261105/supplemental_benchmarks.md) · [XNLI提示敏感性开发对照](research/aya_4b_20261105/xnli_format_diagnosis.md)
- [翻译解码诊断](research/aya_4b_20261105/translation_decoding_diagnosis.md) · [修正候选的冻结计划](research/aya_4b_20261105/translation_repair_candidate_plan.md)

有阅读/摘要收益、SSU小幅知识收益和部分通用任务退化；翻译仍存在严重问题，尚未声称修复。旧数学路线B及取消方法保持停止。下方“进行中”等表述属于历史快照，本轮当前状态以上述入口为准。
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

