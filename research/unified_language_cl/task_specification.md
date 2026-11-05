> 原始研究任务书，用于追溯范围与验收要求。实际数据资格、执行设置与已完成结果以冻结协议和当前报告为准；本轮只执行 ES→TH，FR第三阶段暂不启动。

# 多语言持续学习：统一语言顺序下的 Single FFT / FFT-SEQ 实验计划

我建议这轮先把目标收窄为：**建立七条任务轨道上可重复、训练与测试匹配的语言增量学习基线。**

不开发 My Method，不运行 SST 或 Replay。我们首先要回答：

1. 在相同任务、不同语言的数据上，Qwen3-1.7B 能否通过 Single FFT 学到新能力？
2. 已经学习第一种语言后，FFT-SEQ 能否继续学到第二种语言？
3. 新语言学习是否会损害旧语言能力？不同任务之间的规律是否一致？

## 一、统一实验语言顺序

**ES**

西班牙语

**TH**

泰语

**FR**

法语

主顺序：Spanish → Thai → French。先完成 ES→TH，第三阶段 FR 暂不启动。

我重新核对了主要数据源的语言覆盖：MathOctopus、MASSIVE、MTOP、WikiLingua、XNLI、XL-Sum 和 SIB-200 均包含这三种语言。MathOctopus 官方明确给出了三语训练数量；WikiLingua 包含三语的部分平行文章；XL-Sum 三语都有 train/dev/test，但不属于严格平行语料。

**重要区别：语言覆盖已得到来源支持，但“足够多的三语共同训练ID”尚未得到文件级核实。** 这必须由 Codex 下载实际数据后检查。

## 二、七条实验轨道到底训练什么、测试什么？

| 轨道 | Single/SEQ 训练数据 | 开发与正式测试 | 主指标 |
|--------|--------|--------|--------|
| **MathOctopus** | MGSM8KInstruct 多语言数学题及推导 | 训练源题内部冻结 dev；MGSM-Rev2、MSVAMP 测试 | 最终数字答案 ACC |
| **MASSIVE** | 官方 train：多语言 utterance→intent | 官方 dev / test | Intent ACC、有效标签率 |
| **MTOP** | 官方 train：utterance→结构化语义解析 | 官方 eval / test | 结构 Exact Match、Intent ACC |
| **WikiLingua** | 官方训练部分：文章→操作指南摘要 | 官方对应划分 dev/test | ROUGE-Lsum、BERTScore（可选） |
| **XNLI** | 多语言翻译 train：前提+假设→3分类 | 官方 validation / test | 3分类 ACC |
| **XL-Sum** | 各语言官方 train：新闻→摘要 | 各语言官方 dev/test | ROUGE-Lsum |
| **SIB-200** | 官方 train：新闻句子→7类主题 | 官方 validation/test | ACC、Macro-F1 |

MathOctopus 是数学推理实验，XL-Sum 和 WikiLingua 是开放式生成实验，其余轨道以分类或结构化输出为主。

我建议第一批优先跑 **MASSIVE、MathOctopus、XNLI、MTOP**，覆盖四种实质不同的任务。WikiLingua、XL-Sum 和 SIB-200 在第一批做完整数据审计与 Base 开发评测，训练可以在第一批确认协议正确后继续。

## 三、所有轨道使用同一种比较结构

第一轮：ES → TH

Single FFT

Base → ES

Base → TH

FFT-SEQ

Base → ES

ES → TH

---

每条轨道只需三个训练段：Single ES、Single TH、从同一 Single ES Final 继续的 TH。两种方法在 ES、TH、FR 和适当的 EN 任务上统一评测。

Same-ID 与 Disjoint-ID 不应该立刻铺开所有七条线。

第一步先建立主顺序学习结果；之后只在具有可靠共同语义ID的轨道上比较这两种模式。

下面是完整的 Codex 执行任务书。


# Codex 科研任务书：统一语言顺序的多任务 Single FFT / FFT-SEQ 基线实验

## 一、项目和研究目标

项目仓库：

https://github.com/lxy308297917-cloud/Multilingual-CL-10.31

本次建立新的多语言持续学习实验体系，独立于此前 Aya SW→Hausa 实验。

**统一语言顺序：Spanish (ES) → Thai (TH) → French (FR)。**

第一轮只运行：

- Single FFT：Base→ES
- Single FFT：Base→TH
- FFT-SEQ：Base→ES→TH

第三阶段FR、顺序反转、Replay、SST和My Method暂不执行。

核心科学问题：

1. 相同任务的不同语言版本能否让Qwen3-1.7B产生明确的语言任务学习？
2. 完成ES训练后，继续TH训练能否获得新语言净学习收益？
3. TH学习时，ES已有能力会下降、保持还是提高？
4. 同一语言顺序下，数学、分类、语义解析和摘要任务的持续学习规律是否一致？
5. Same-ID与Disjoint-ID任务内容的差异是否影响后续语言学习？此项放在主要基线完成后。

本轮**不开发任何新优化算法**。

不得为了得到正向效果而反复根据正式test更换语言、学习率、checkpoint或评分方式。

---

## 二、实验模型

基础模型保持原项目锁定版本：

`Qwen/Qwen3-1.7B`

revision：

`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

注意，这是官方后训练版本，不是Qwen3-Base。

所有轨道从相同Base独立初始化，禁止跨轨道复用微调权重。

初始训练协议：

| 项目 | 设置 |
|---|---|
| 微调方式 | Full Fine-Tuning |
| 框架 | 现有LLaMA-Factory + DeepSpeed |
| Precision | BF16 |
| Assistant-only | 是 |
| Thinking | 关闭 |
| Optimizer | AdamW |
| LR | 1e-5 |
| Weight decay | 0 |
| Effective batch | 16 |
| Scheduler | Cosine |
| Warmup | 5% |
| Seed | 42 |
| Packing | False |
| Cutoff | 默认2048；截断严重时先审计，不直接修改正式协议 |
| Checkpoint | 固定步数保存，Final作为主要结果 |

第二阶段从第一阶段相同Final权重启动，重新初始化Adam、scheduler，记录完整训练状态。

第一阶段和第二阶段的每个checkpoint必须有唯一名称、资产哈希和配置ID。

由于数学推导、摘要、语义解析的长度差异较大，不允许仅因为样本数量相同就声称计算量和监督token完全匹配。逐轨道报告真实token和GPU成本。

---

## 三、实验轨道及数据来源

必须审计以下七条轨道。

### Track A：MathOctopus——数学推理

训练：

https://huggingface.co/datasets/Mathoctopus/GSM8KInstruct_Parallel

测试：

https://github.com/google-research-datasets/MGSM-Rev2

https://huggingface.co/datasets/Mathoctopus/MSVAMP

任务定义：

输入目标语言的数学问题，输出数学解题过程及最终数值答案。

候选语言：ES、TH、FR。

官方训练数据量约为ES7470、TH7473、FR7469，MSVAMP每语言1000题，MGSM-Rev2每语言250题。

必须核查：

- 各语言真实样本数量。
- 训练源题ID是否可靠对齐。
- 推理文本的目标语言是否正确。
- 最终数值答案是否一致。
- 与MGSM、MSVAMP是否有题目泄漏。
- 参考推导是否能够直接作为assistant监督。
- Qwen3-1.7B在开发题上的数学能力是否处于严重地板。

从训练源题另行划分共同语义ID的开发集。

测试MGSM-Rev2与MSVAMP只允许在冻结协议后正式运行。

主要指标：最终数值ACC；辅助指标为答案解析率、严格与宽松评分及输出语言。

### Track B：MASSIVE——多语言意图识别

官方数据：

https://huggingface.co/datasets/AmazonScience/massive

候选：`es-ES`、`th-TH`、`fr-FR`。

每语言官方train11514、dev2033、test2974。

先只训练intent classification：

User输入：目标语言utterance。

Assistant输出：统一的官方intent标签。

不在本轮同时训练slot，以免把两种不同输出任务混合。

主要指标：

- Intent Accuracy
- Invalid Label Rate
- Macro Accuracy

主生成评分和受限候选标签评分分别报告，提前冻结主指标。

必须用官方共同ID核对多语言标签一致性。

### Track C：MTOP——组合语义解析

论文：

https://aclanthology.org/2021.eacl-main.257/

候选：ES、TH、FR。

训练输入为多语言任务型utterance，输出为标准化的语义解析结构。

优先使用论文的compositional decoupled表示，并在实际数据审计后冻结解析序列化规则。

主要指标：

- 完整语义结构Exact Match
- Intent Accuracy
- Slot F1（可实现时）

MTOP仅为近似平行数据，不能假设所有语言都具有相同source ID。必须实际核对共同题目、过滤记录与标签结构。

若合法可用的共同ID少于预期，按实际数量降低统一预算，并报告原因。

### Track D：WikiLingua——生成式操作指南摘要

数据：

https://huggingface.co/datasets/esdurmus/wiki_lingua

候选：ES、TH、FR。

输入操作指南文章，输出对应语言的摘要。

使用官方或作者明确支持的防泄漏数据划分。

必须核查：

- 文章和图片对齐关系。
- 同语义文章的语言交集。
- 跨语言文章是否跨train/test泄漏。
- 每语言真正可用的训练数量。
- 长文本截断程度。
- 摘要语言是否正确。

主要指标：

- ROUGE-Lsum
- ROUGE-1/2
- 输出长度与语言正确率

如果没有可靠共同语义ID，不得运行声称严格平行的Same-ID实验；可以先运行各语言原生文章的任务匹配顺序SFT。

### Track E：XNLI——自然语言推断

数据：

https://huggingface.co/datasets/facebook/xnli

候选：ES、TH、FR。

训练输入：Premise + Hypothesis。

Assistant输出固定标签：

`entailment` / `neutral` / `contradiction`

使用多语言翻译train，官方validation和test。

需明确训练翻译来自英文MultiNLI源语义ID，不将其误称为全人工译文。

必须检查：

- 392702个训练源语义ID及各语言翻译字段。
- 跨语言标签一致性。
- 训练/dev/test对应源题隔离。
- 三分类标签是否出现语言偏好。
- Base是否具有过高的初始ACC。

主指标：三分类ACC；辅助为Macro-F1及非法标签率。

### Track F：XL-Sum——真实新闻摘要

数据：

https://huggingface.co/datasets/csebuetnlp/xlsum

候选：ES、TH、FR。

各语言都有官方train/dev/test，但语义文章通常不平行。

训练输入：目标语言新闻文章。

Assistant输出：同语言摘要。

主要指标：ROUGE-Lsum，同时报告生成长度、参考长度和语言正确率。

注意：本轨道主要研究语言与新闻内容同时变化的现实顺序SFT。

不声称它能够严格隔离语言切换效应，也不构造伪造的Same-ID对应。

### Track G：SIB-200——新闻主题分类

数据：

https://huggingface.co/datasets/Davlan/sib200

候选：`spa_Latn`、`tha_Thai`、`fra_Latn`。

每语言train701、validation99、test204。

输入新闻句子，输出固定7类主题之一。

主要指标：

- Accuracy
- Macro-F1
- Invalid Label Rate

使用官方`index_id`核对共同语义ID。

此轨道样本量远小于其他任务，必须采用单独的小样本训练预算，不得强行凑成3200条或重复样本冒充独立数据。

---

## 四、P0：全轨道数据审计——必须首先完成

七条轨道全部执行，不允许跳过。

建立统一审计表：

- Dataset、revision、license
- ES/TH/FR train/dev/test数量
- 共同源语义ID数量
- 单语言有效样本数量
- 重复/损坏/空答案数量
- 输入及assistant监督token分布
- 标签、答案或摘要质量检查
- 最大长度及截断比例
- train/dev/test泄漏检查
- Base开发评分
- 正式test是否被历史研究曝光

对于能够可靠建立共同语义ID的数据，所有语言必须按语义ID共同分割，禁止同一个语义问题的不同翻译分别出现在训练和开发/测试集合。

对数学特别注意：

答案相同不代表题目相同；列表顺序相同也不自动证明源题相同。

P0的成果是判断各轨道是否可以开展训练，并明确其局限。

如果某个轨道的官方数据无法合法取得、缺乏必要答案、存在不可接受的split污染，标记BLOCKED，不使用未经许可或未经审核的第三方数据替代。

---

## 五、P1：Base初始能力与评分工具验真

对每条通过P0的轨道，使用相同Qwen3 Base运行ES、TH、FR、EN（适用时）的开发评分。

需要：

1. 固定chat template。
2. 固定目标语言任务指令模板。
3. 分类任务检查标签解析与受限候选概率。
4. 数学验证数值答案提取。
5. 结构化任务检查解析树合法性。
6. 摘要验证ROUGE计算及语言输出。
7. 输出逐任务开发准确率或任务指标。
8. 检查天花板、地板、极强标签偏置等问题。

P1只使用dev，不用正式test决定语言顺序或训练参数。

开发评分很高不等于该轨道无价值，应明确报告可提升空间；开发评分很低也不能直接断言模型没有任务能力。

---

## 六、P2：两语言Single FFT与FFT-SEQ

优先顺序：

1. MASSIVE
2. MathOctopus
3. XNLI
4. MTOP
5. WikiLingua
6. XL-Sum
7. SIB-200

可先完成前四条轨道，再根据实际GPU资源推进剩余轨道；不要因为某轨道无正收益而隐藏或删除它。

### 每轨道训练三个训练段

A. Single ES：

`Base → ES`

B. Single TH：

`Base → TH`

C. FFT-SEQ：

`Single ES Final → TH`

C必须复用A的最终权重，不能重新训练另一个不同的ES起点。

### 默认数据预算

对于存在足够训练数据的轨道：

- 第一阶段3200个训练实例。
- 第二阶段3200个训练实例。
- Effective batch16。
- 每阶段200 optimizer steps。
- 同一个seed42和冻结数据顺序。

如果轨道具有共同语义ID，第一轮优先使用Same-ID配置。

对于共同ID不足的轨道，以P0通过的实际规模为准，冻结预算；不得通过重复样本伪造3200个独立实例。

SIB-200单独使用最多640条train/语言，对应40步；预留其余训练样本不得加入dev/test。

保存初始模型、step100及step200 Final；短轨道按其合法训练步数确定保存点。

正式主结果固定Final，不根据正式test选择最佳checkpoint。

### 评测矩阵

每条轨道生成：

| Model | ES | TH | FR |
|---|---|---|---|
| Base | score | score | score |
| Single ES | score | score | score |
| Single TH | score | score | score |
| FFT-SEQ ES→TH | score | score | score |

并另外报告EN该任务能力及项目已有通用能力的适当子集。

第一语言收益：

G_ES = S_ES(W1) - S_ES(W0)

第二语言收益：

G_TH = S_TH(W2) - S_TH(W1)

第一语言遗忘：

F_ES = S_ES(W1) - S_ES(W2)

第一阶段对未见TH的迁移：

T_TH = S_TH(W1) - S_TH(W0)

每项指标必须保持原始任务单位，不能简单将ACC、ROUGE和数学分数平均为一个不加说明的综合得分。

### 开发集统计

对于严格平行任务，按共同语义ID进行配对统计。

对于XL-Sum非平行任务，各语言独立报告统计区间，不假装不同新闻文章之间存在逐题配对。

固定模型的题目bootstrap区间不等于不同训练种子的方差。

---

## 七、P3：Same-ID与Disjoint-ID受控比较

在P2确认训练、数据和评价正确之后，优先对MASSIVE、MathOctopus、XNLI开展。

Same-ID：

ES和TH使用同一批3200个源语义ID，各自使用目标语言版本。

Disjoint-ID：

ES使用一批3200个源语义ID，TH使用另一批3200个不重叠的ID。

两组都使用相同任务定义、相同阶段预算及独立train来源。

Disjoint-ID须有至少6400个有效共同训练语义ID。

若共同ID不足，提前冻结更小的预算，不得事后根据模型分数调整样本。

这一比较不能直接解释为纯语言因果，因为Same-ID存在对题目内容的重复学习，而Disjoint-ID同时引入新内容。

因此必须分别讨论语言切换效应、重复内容效应与任务难度差异。

P3可安排为第二批，不要求在第一批GPU训练期间全部执行。

---

## 八、P4：第三语言扩展——暂不自动运行

候选：

`ES → TH → FR`

前置条件：

- 两语言配置完整。
- 数据和评分验真通过。
- 至少一条轨道能在开发指标上观察到可信的新语言学习。
- 研究者明确冻结三阶段预算和数据分割。

三语言Same-ID、Disjoint-ID需要重新冻结三阶段共同预算，例如每阶段2000个源语义ID、125 optimizer steps；不能把先前3200条的ES→TH训练直接拼接为另一个预算的三阶段实验。

第三阶段需要额外Single FR正对照。

不因个别任务没有遗忘而人为制造遗忘。

---

## 九、必须保留的旧能力监测

新轨道的主任务得分是核心，但仍要监测模型的一部分通用能力。

建议使用原项目冻结协议，至少记录：

- EN IFEval prompt strict
- EN GSM8K strict及flexible
- EN阅读或知识
- 当前轨道的EN同任务指标

如果由于GPU成本无法每个中间checkpoint全部评测，至少保证Base和各阶段Final有统一的旧能力记录。

数学轨道要特别警惕数学任务增益只是格式改善。

分类轨道要特别警惕答案标签先验变化。

摘要轨道要同时报告ROUGE、输出长度和语言正确率。

不要把所有英语指标笼统称为一个“EN Forgetting”数字。

---

## 十、执行顺序与资源

第一批：

完成七轨道P0，以及MASSIVE、MathOctopus、XNLI、MTOP的P1/P2。

第二批：

WikiLingua、XL-Sum、SIB-200的P1/P2，以及通过条件的Same-ID/Disjoint-ID补充。

每次训练前：

- 核查GPU归属。
- 核查模型与数据资产哈希。
- 核查已有任务是否正在执行。
- 禁止覆盖旧结果。
- 使用独立输出路径。
- 检查保存空间。
- 记录实际输入/监督token和GPU资源。

不得为了追求实验数量牺牲输出正确性、训练可复现性或数据来源核验。

---

## 十一、交付

建议建立：

`research/unified_language_cl/`

包含：

- `README.md`
- `dataset_audit.md`
- `language_alignment.md`
- `evaluation_protocol.md`
- `single_fft_results.md`
- `fft_seq_results.md`
- `stage_language_matrix.md`
- `same_vs_disjoint.md`
- `cost_report.md`
- `research_conclusion.md`

对应代码与结果放入：

- `scripts/unified_language_cl/`
- `configs/unified_language_cl/`
- `results/unified_language_cl/`

每条轨道要有独立子目录、独立实验配置、唯一checkpoint ID及机器可读输出。

完成之后提交GitHub，提供提交SHA。

最终报告必须回答：

1. 哪些任务上的Single FFT确实学到了目标语言任务能力？
2. 哪些任务上的FFT-SEQ第二阶段获得了新的净收益？
3. 哪些任务出现了旧语言遗忘？哪些没有？
4. 新语言收益与旧语言下降是否存在稳定的学习—保持权衡？
5. Same-ID和Disjoint-ID是否呈现不同规律？
6. 数学推理、语义理解、结构解析和摘要是否表现出不同的CL难度？
7. 哪一条轨道最适合未来开展Replay、SST和My Method对照？
8. 哪些结果属于数据/任务/读出问题，而不是已经确认的参数遗忘机制？

**停止条件：** 不要求每条任务必须产生正向收益。若没有可靠学习信号，先报告任务地板、天花板、数据不匹配或优化限制，不得通过正式test反复调参挑成功结果。

**本轮完成标准：建立可信的Single FFT与FFT-SEQ实验体系，而不是证明某个新算法有效。**


---

## 四、我最希望这轮得到什么结果？

最理想的不是七条轨道全部出现遗忘，而是它们能够提供**有解释力的差异**。

例如，MASSIVE 的第二语言任务很容易学会且遗忘较小，MathOctopus 的第二语言数学收益明显但第一语言数学下降，XL-Sum 又表现出不同的生成式遗忘模式。

这样的差异能让我们进一步判断：语言持续学习的困难，究竟更受语言切换影响，还是更受任务复杂度、输出形式与数据内容影响。

目前尤其应该关注 **MathOctopus**。因为它直接对应著名的 GSM8K 数学推理任务，也与 Qwen3 的生成式指令微调更接近。MASSIVE 可以充当严格的工程和语言增量学习对照，MTOP、XNLI、摘要则检查规律能否跨任务成立。

我的执行建议是：先让 Codex 完成七轨道的数据审计与 Base 开发评测，再优先启动 MASSIVE 和 MathOctopus 的 Single ES、Single TH、FFT-SEQ ES→TH。

等这两条出现清楚的阶段学习曲线后，再决定是否值得把训练资源扩展到全部七条线。这样既不会遗漏你看好的 benchmark，也能避免重演此前 Aya→Hausa 中“花了大量 GPU，但第二阶段学没学会仍不清楚”的问题。
