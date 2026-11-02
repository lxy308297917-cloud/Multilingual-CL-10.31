# 最新交付：Benchmark v2

## Benchmark v2 补充测评（2026-11-02完成）

固定Base/FFT/LoRA/SSU Final，全量生成60,352条；16组翻译/摘要正式评分及8组代码生成/静态检查完成，28项测试通过。SW摘要和EN→SW有收益，SW→EN下降，EN摘要没有一致遗忘。代码pass@1为NOT_RUN；CrossSum仅可行性审计。原机制诊断负面结果保留。

- [Benchmark v2报告](bench_v2/benchmark_report.md)
- [正式表与配对区间](../results/bench_v2/summary/paper_table.md)
- [复现与限制](bench_v2/reproduce.md)

---

# 给 ChatGPT 的项目交接：Qwen3 多语言 SFT 与机制诊断

更新时间：2026-11-02。本轮主要实验、报告及代码结果交付已完成。请先审查现有证据，再讨论下一步；当前没有已经验证的新方法。

## 1. 我们希望你协助讨论什么

我们最终希望设计一种多语言持续学习方法，兼顾目标语言收益与通用任务保留。目前先检验：SW SFT 是否存在可利用的语言接口／共享任务功能部分分离机制。请基于仓库证据判断假设是否值得继续，以及还缺什么最有区分力的证据。不要假设所有英语任务一起遗忘，也不要把CPT结论直接迁移到指令SFT。

## 2. 已完成的 baseline

- 官方后训练 `Qwen/Qwen3-1.7B`，revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，不是另一个Qwen3-Base。
- Aya Collection SW6400，1 epoch，400 optimizer steps，有效batch16，assistant-only，non-thinking；固定Final，不依据正式测试选checkpoint。
- 输入2,884,423 tokens，移位回答监督1,078,347 tokens；长度2048，不packing。
- FFT LR1e-5；LoRA r16/alpha32、LR1e-4；SSU-Wanda静态英语保护、LR1e-5。SSU保护40.9594%唯一参数，校准含数学偏置，不是历史SSU15%。跨方法配方不同，不能把差异全部归因于机制。

|正式主要指标（%）|Base|FFT|LoRA|SSU-Wanda|
|---|---:|---:|---:|---:|
|SW Belebele ACC|28.33|36.67|36.89|35.44|
|SW Global-MMLU ACC|27.87|31.65|31.38|31.24|
|EN Belebele ACC|60.78|77.11|78.22|79.67|
|EN MMLU ACC|57.01|58.45|58.33|58.82|
|IFEval prompt strict|66.91|27.17|31.05|47.50|
|GSM8K strict|45.03|3.79|7.35|13.80|
|GSM8K flexible|73.01|61.26|56.79|65.43|

完整80项seed42指标含双向翻译、IFEval其他标准指标等，见 [CSV](../results/baselines_seed42.csv) 与 [JSON](../results/baselines_seed42.json)。旧MURI B的训练数据、预算、协议不同，应独立看待，不与新Aya轨道拼表。已有完成复测保留，不继续新增训练种子。

## 3. 已完成的机制实验

1. 三模型有效参数差分：正确PEFT merge/scaling与tied参数计数，逐28层/组件各280行；只作描述。
2. 同60输入的Base/Final激活差分，各2352行；prompt与teacher-forced参考分开，不假装跨语言token对齐。
3. 独立232条开发输入，17个预定干预加Base/Final，共19条件/模型、13,224条预测；代表层0/14/27，Attention/MLP/Block、连续层组、恢复、注入、alpha0.5、匹配参数量／范数随机对照。不是全层搜索。
4. 三组共1224行条件效应、72行交互；同语义pair/重复提示聚类，2000次配对bootstrap，未做多重校正。
5. 同输入激活补丁最终v3，各26样本/368行，包括随机扰动、26个中性语义与14个不同题控制。identity误差0，读数为参考答案LL，不是生成准确率。
6. 37项CPU测试通过；直接在公开副本再运行同样37项也通过；真实模型smoke、identity、参数回滚及随机对照匹配均通过。

## 4. 当前结论与关键限制

**已经看到的现象**：SW阅读与知识提升，EN阅读与知识也提升，数学/指令等部分通用任务退化。SSU在正式IFEval/GSM8K上减轻一部分退化，但未恢复Base。

**尚未获得的证据**：这批少量预定模块，没有可靠、清晰的局部功能分离结果。三组各192个字面恢复效应单元，经验95%区间均包含0。这个结果不证明模块无功能，也不否定更细粒度或其他范围下的分离。

**必须注意的比较基准**：恢复用 `Final−restore(G)`；单独注入的充分性应看 `inject(G)−Base`。`Final−inject(G)`移除了其余全部更新，不能当作G的贡献。FFT/SSU完整Block还包含有变化的LayerNorm，Attention/MLP/Block差值不是纯交互。

**评分限制**：原数学受限答案提取并非真正内容准确率。事后保守数值诊断中，16题的可解析正确/错误/未知分别为Base12/2/2、FFT2/11/3、LoRA0/6/10、SSU7/5/4。未知不判错，不改正式评分；FFT确有错误数字，因此不能说所有退化都来自格式。LoRA未知多，不能说全部答错。

**开发材料限制**：阅读span不是Belebele，合成约束不是官方IFEval，0-shot数学不是正式5-shotGSM8K。知识44行只有22语义实例；EN合成约束16行只有13提示簇。SSU在这批合成指令为5/16，FFT6/16，与正式IFEval顺序不同，应保留而不是只展示有利任务。短答alias有地板、SW合成措辞未独立人工校验，自由文本输出语言未充分测量。

**激活限制**：相对L2受Base每层尺度影响，不能把中间低谷解释为语义hub保留；补丁LL不等于生成收益，中性控制可能分布外。没有可靠跨语言token对齐，本轮未做位置一一的跨语言补丁。26例分属13个任务／语言方向组，每组仅2例；中性语义和不同题控制未匹配扰动范数，不能将对照差异单独解释为语义作用。

**事后选项轮转补充**：SW22题中两次都正确由Base2题变为FFT/LoRA各13题、SSU10题；比只看字母分布更稳，但不能区分新知识、语言理解或作答校准。参见[完整补充](mechanism/choice_rotation_audit.md)，不是新的正式选模指标。

## 5. 建议你重点审查的四个问题

1. 当前证据究竟更支持输出行为／指令习惯变化，还是任务求解功能变化？哪些尚不能区分？
2. 本轮弱局部干预能否对分离假设形成有意义约束，还是主要受样本量、地板、离散评分和干预范围限制？
3. 如果只补充一个最小实验，什么设计最能区分竞争解释？请明确开发材料、控制组、读数和可能否定假设的结果，避免全层穷举。
4. 哪些结论已经足以指导方法方向，哪些仍不足以支持新方法？请不要为了提出My Method而强行归因。

## 6. 推荐阅读顺序

1. [完整机制报告：Q1–Q4、正式与开发结果、限制](mechanism/mechanism_report.md)
2. [诊断结果与图表入口](../results/mechanism/README.md)
3. [独立探针来源和质量](mechanism/development_probes_v1.md)
4. [实验协议](mechanism/experiment_protocol.md) 与 [资产清单](mechanism/asset_inventory.md)
5. [层级参数描述补充](mechanism/parameter_summary.md)
6. [代码与命令](mechanism/README.md)，重点 `weights.py`、`module_swap.py`、`analyze_interventions.py`、`activation_patching.py`、`diagnose_math_answers.py`。

模型权重、原始数据文本和大日志未上传；公开副本提供来源/选题ID与科学统计，外部重跑还需取得模型、数据及few-shot排除材料。用户取消的paired/source/shuffled/replay四组训练及其测评保持停止，不是当前研究依据。
