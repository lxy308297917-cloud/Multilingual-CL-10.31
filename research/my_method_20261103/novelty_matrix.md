# 创新性排重矩阵

核查范围：下列指定版本的方法正文、公式与已注明的官方源码。不是穷尽检索；没有实际复现的论文不标成已复现。“无旧数据”指无需历史原始样本，仍可能保存特征统计、梯度或参数。诊断论文的训练控制项写“不适用”。

## 科学问题、信号与约束

| 方法/版本 | 科学问题与方向识别信息 | 历史样本需求 | 约束/适配对象 | SVD与主方向保护 |
|---|---|---|---|---|
| [SST v1](https://arxiv.org/html/2610.22293v1) | 适配新任务时保留原始谱结构；参考权重谱 | 无；保存参考/基 | 优化器之后的累计实际权重变化 | 精确SVD；保护top-k，比例是rank而非能量 |
| [OPLoRA v1](https://arxiv.org/html/2510.13003v1) | 低秩适配不改原奇异三元组；权重谱 | 无 | 前向的双侧投影低秩分支 | SVD；固定主方向，非仅初始化 |
| [Sculpting v1](https://arxiv.org/html/2504.07097v1) | 任务相似度与逐层可塑性；新旧输入/输出信号 | 相似度需旧任务信息/代表统计；不能称纯权重无数据信号 | v1梯度式及分解训练；现行OSFT源码另列 | 每任务SVD；v1式G−P_U G P_V允许交叉块 |
| [MiLoRA NAACL2025](https://aclanthology.org/2025.naacl-long.248.pdf) | 小谱成分是否更适合作适配初始化 | 无 | 可训练A/B，初始化后可旋转 | SVD小方向初始化；无全程严格主空间保护 |
| [PiSSA v1](https://arxiv.org/pdf/2404.02948v1) | 主谱初始化提高低秩适配效率 | 无 | 可训练主成分A/B，冻结残差 | SVD主方向用于学习，非保护 |
| [CorDA v1](https://arxiv.org/html/2406.05223v1) | 任务/知识上下文定向分解；输入协方差 | 需任务或知识校准数据，不必是旧训练原始集 | W与上下文协方差组合的分解/初始化 | SVD(WC)，KPA/IPA选择不同成分；初始化非永久正交 |
| [O-LoRA EMNLP2023](https://aclanthology.org/2023.findings-emnlp.715.pdf) | 顺序任务减少低秩空间干扰；旧A因子 | 无原始回放；保存旧因子 | 新旧因子的软正交惩罚 | 不以主谱SVD为必要条件；不保护固定top-k |
| [LoDA ICML2026](https://proceedings.mlr.press/v306/he26f.html) | 共享/隔离两支复用旧知识；新旧特征Gram | 保存旧特征统计，不是默认旧样本回放 | 特征子空间选A、分阶段更新B及合并校准 | 特征Gram/广义特征问题；不按权重top-k保护 |
| [GPM ICLR2021](https://arxiv.org/pdf/2103.09762v1) | 不回放原样本也保持旧表示；激活记忆 | 保存旧激活基，无需持续保存原样本 | 梯度在旧激活基的补空间 | 激活SVD；不是权重主谱 |
| [OGD AISTATS2020](https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf) | 一阶保持旧预测；旧输出Jacobian | 基由旧样本构建，后续保留输出梯度记忆 | 投影新loss梯度 | Gram-Schmidt为主；非权重top-k |
| [TL-CL EMNLP2024](https://aclanthology.org/2024.emnlp-main.676.pdf) | 顺序增加任务和语言；语言/任务adapter、Fisher重要性 | 主方法无需旧样本回放；ER对照另有buffer | adapter与EWC式保持 | 非权重谱SVD保护 |
| [LayRA v1](https://arxiv.org/html/2509.11414v1) | 无原始预训练数据时连续加语言；初末层语言接口假设 | 新语言CPT数据，不回放原预训练集 | 选层LoRA、语言向量与合并 | 非SST主谱约束 |
| [Two-Phase CFT ACL2026](https://aclanthology.org/2026.findings-acl.1595.pdf) | 英语IFT→多语言IFT为何遗忘；指令数据相似性 | 不同实验涉及英语/多语言数据、生成回放 | 数据组织、层冻结与回放控制 | 非谱SVD方法 |
| [Semantic Hub CPT ACL2026](https://aclanthology.org/2026.findings-acl.1218.pdf) | 多语言CPT的层功能位置；跨语言检索、层交换 | 机制分析需跨语言材料 | 表征诊断及参数/层干预 | 不以SVD确定训练约束 |
| [MWork v1](https://arxiv.org/pdf/2402.18815v1) | 多语言理解/求解/生成的内部机制；神经元概率与消融 | 需多语言诊断样本 | 神经元/计算阶段干预 | 非谱方法；不是顺序CL算法 |
| [Language-Specific Neurons…2025](https://aclanthology.org/2025.insights-1.6.pdf) | 语言神经元选择是否改善跨语言迁移；LAPE与干预 | 需语言样本/任务训练材料 | 神经元屏蔽、masked LoRA | 非谱方法；结论有模型/任务范围 |

## 复用、顺序范围、共享语义、部署与实质差异

“语义”列区分跨语言机制分析与**显式同语义功能统计参与更新规则**；不能把二者混为一种方法。

| 方法 | 旧方向复用与多阶段 | 跨语言共享任务语义 | 新增参数/推理身份或路由 | 对候选A的约束 |
|---|---|---|---|---|
| SST | 固定主方向禁止改，尾空间仍可复用；刷新支持多任务 | 非显式语言条件规则 | 全量训练；合并单模型，无任务路由 | SST加语言数据不新；后学尾方向可能仍被覆盖 |
| OPLoRA | 固定双侧自由空间，允许其中多任务更新；不自动保护后来尾方向 | 非显式 | 新低秩分支可合并，无必需路由 | 正交分支及主三元组保持已有 |
| Sculpting/OSFT | 按任务相似度分配可塑性，多阶段 | 一般任务信号，不等于同语义多语言配对 | 现行低谱因子可重构权重 | 数据感知rank及修正Adam旋转已有 |
| MiLoRA | 因子训练中可移动；原论文不是完整语言顺序方案 | 无显式 | LoRA参数，可合并 | 小/大/随机谱初始化已有对照 |
| PiSSA | 允许主成分学习；多阶段保持不是原保证 | 无显式 | LoRA式因子，可合并 | 不能预设所有主方向都该冻结 |
| CorDA | 上下文模式控制适配/知识保持；不等于无限阶段记忆 | 数据感知，不是配对语义梯度规则 | 因子可合并 | 用旧知识数据挑谱空间已有 |
| O-LoRA | 新因子避开旧空间，旧因子保留；支持顺序 | 一般任务，不是显式跨语言语义规则 | 旧adapter数量/存储随阶段增长，合并与部署策略须注明 | 正交历史更新与语言移植都不足为创新 |
| LoDA | 共享分支复用，隔离分支新学；顺序类别增量 | 原实证为视觉类别，不是Qwen多语言SFT | 任务期分支与合并，无推理任务ID依赖的主张须按版本核查 | 共享/私有空间不能作为新贡献 |
| GPM | 旧记忆内通常禁止更新，补空间学习；顺序任务 | 一般任务，无显式跨语言配对 | 无必需推理路由；基底记忆可增长/饱和 | 激活相关保护及记忆容量问题已有 |
| OGD | 保持旧输出的一阶方向；顺序任务 | 一般任务，可用于语言但原法非语言专设 | 梯度记忆增长，无推理路由 | 功能梯度保护本身已有 |
| TL-CL | 复用任务/语言因子，新增语言/任务多阶段 | 显式任务与语言因子，不是本文配对梯度统计 | O(T+L)adapter、任务/语言选择 | 语言与任务分解、阶段增长必须匹配对照 |
| LayRA | series/merge连续加语言，保留中层 | 语言角色假设与跨语言下游，不是同义梯度门 | 选层LoRA可合并；合并策略不同 | 语言接口/层选择不是新的普遍机制 |
| Two-Phase CFT | 两阶段顺序语言IFT，回放允许旧任务更新 | 对指令任务相似性有直接控制 | 主模型全量/层冻结，无新路由必需 | 本项目source配额相同不代表语义项目相同，仍有任务难度混杂 |
| Semantic Hub CPT | 诊断CPT迁移，不等价于本文顺序控制器 | 检索与层干预直接研究跨语言共享 | 诊断不自动新增部署参数 | 相似度/漂移不能替代任务ACC；CPT不能直接推SFT |
| MWork | 非顺序更新算法，复用/多阶段项不适用 | 提出理解—求解—生成机制假设 | 消融/分析，不是新部署框架 | 不能预设Qwen3中层是英语 |
| Language-Specific Neurons… | 干预/迁移研究，不是长序列CL保证 | 直接研究跨语言迁移，结论限被测设置 | masked LoRA可合并，选择神经元需语言样本 | 负证据须保留，不能靠“语言神经元”名称建立创新 |

## 最近邻补充：直接否定“简单组合即创新”

| 原文 | 已覆盖部分 | 仍需验证的差异 |
|---|---|---|
| [FS-DGPM](https://arxiv.org/html/2110.04593v1) | 新旧梯度沿记忆基兼容时释放、冲突时收紧 | 语义配对是否提供超出已有兼容性门的信息 |
| [SVFT](https://arxiv.org/html/2405.19597v1) | 固定奇异向量外积的可训练稀疏系数 | 已学Δ而非W0的功能语义选择；系数化本身不新 |
| [FCCA LLM](https://arxiv.org/html/2609.00762v1) | 有符号输入—误差交叉矩、Fisher白化、固定小核；RawGrad对照 | 同语义EN/SW系数梯度跨协方差的额外价值；本文未白化，不借用其最优性证明 |
| [FCCA ASR](https://arxiv.org/html/2609.29800v1) | 跨层共享/私有固定空间、固定总预算rank分配，多语言ASR | 语言顺序与配对语义不是单纯更换ASR任务，但要有实证差异 |
| [UGP](https://arxiv.org/html/2607.11163v1) | 语言平衡的A-GEM参考方向与回放，ASR | 不能把语言均衡梯度保护改名 |
| [COMPASS](https://arxiv.org/html/2604.20720v1) | 语言/域适配、语义聚类与路由 | 其使用未标注评估/使用分布的聚类选择，与本项目test隔离协议不同 |
| [PaLoRA](https://arxiv.org/html/2610.04226v1) | 历史更新SVD压缩、nullspace、rank相关节奏、有限精度泄漏假设 | 本项目BF16参数舍入的实测并不自动构成新理论或方法 |
| [Learning Dynamics](https://arxiv.org/html/2609.33620v1) | 同语义跨语言行为互动、readout与backbone通道 | 不能将跨语言功能互动当作未被研究；需实际选择规则与matched控制 |
| [A Deafening Silence](https://arxiv.org/html/2610.09835v1) | 输出稀有token的Adam效应、head epsilon及行控制 | 本项目需下游ACC而非旧域CE证明，且LoRA head不变仍有代码遗忘 |
| [MAD-X](https://aclanthology.org/2020.emnlp-main.617.pdf) | 输入/输出可逆语言适配及任务语言组合 | 简单输入输出角色分解/语言adapter不新 |

## 版本纠正与未读部分

- Sculpting v1式4与现行mini_trainer `fd5b552177bde9a551d30e73660d6a96205c97c0`不能混写。现行源码在Adam前投影梯度、之后再投影低谱因子参数；不能说它没有处理Adam方向变化。ICLR2026 PDF遇403，未核读该版全文。
- LoDA代码 `e34d169189933a63600f37aa78b21864c08a7b59` 的task0训练共享A/B；task≥1的ours分支才固定A而训练相应B。论文Algorithm1描述与这段首任务实现需分开记录，不能说当前代码所有阶段都固定A。
- CorDA++、InfLoRA、LoRA-Null等尚未完成本轮逐公式核读。不得据当前清单声称创新性已成立。

## 第二轮直接近邻：配对、多视图和功能条件控制

下表为方法正文核读；没有运行其完整官方实验，不称严格复现。它们与原16篇共同约束候选创新性。

| 原文/版本 | 已有方法与实际对象 | 数据/顺序/部署边界 | 对当前候选的限制 |
|---|---|---|---|
| [CLO ACL2025](https://aclanthology.org/2025.acl-long.734.pdf) | 翻译英语SFT+目标SFT及跨语言DPO；只训练Attention | 英语/目标各6400及参考模型；不是只有目标语言CE | 语义配对与语言偏好联合保护已有；要匹配英语/翻译/参考计算成本 |
| [CC-Tuning ACL2025](https://aclanthology.org/2025.acl-long.933.pdf) | 平行EN/非EN的FFN激活融合，训练DecisionMaker；推理TransformMatrix最小二乘拟合 | 依赖平行材料，TransformMatrix用1000对材料闭式拟合；不是梯度训练该矩阵 | 共同语义驱动功能对齐已有；不把学习激活融合当新点 |
| [Preserving Cross-Linguality2021](https://aclanthology.org/2021.repl4nlp-1.8.pdf) | GEM保护旧MLM/XSR等跨语言任务 | mBERT/XLM-R及Europarl平行样本，与Qwen生成SFT不同 | 跨语言平行功能约束本身不是空白；GEM对照应匹配旧数据 |
| [PRSR v1](https://arxiv.org/pdf/2610.23807v1) | 平行组均值、语言残差SVD，最后token去残差 | XNLI1500组/15语言，推理表示处理，非SFT顺序训练 | 语义共享与语言残差分离已有；q=K−1是其选择，不是一般数学rank界 |
| [PICO EMNLP2026](https://aclanthology.org/2026.emnlp-main.528.pdf) | −gθ的utility EMA/sigmoid坐标门，周期leading-σ监控 | CL训练；结构标志抑制加噪，原保护梯度仍更新；不是SST完整SVD约束 | 重要性/结构条件的自适应更新门不能单独称创新 |
| [Sequential Reptile v1](https://arxiv.org/pdf/2110.02600v1) | 内轨迹顺序混合任务、一阶meta更新鼓励跨任务梯度对齐 | 多任务meta过程与额外轨迹计算，不等于独立阶段SFT | “促进跨语言梯度对齐”已有，多语言移植仍需实质机制区别 |
| [Share v1](https://arxiv.org/pdf/2602.06043v1) | 堆叠/中心化旧LoRA A/B，SVD共享基、任务系数、临时扩基再整合 | 轻量任务系数仍随任务数增长；Share-full另需旧数据 | 共享谱基+小系数不新；因子受gauge影响，有效Δ改写本身亦需排重 |
| [ZipperLoRA v1](https://arxiv.org/pdf/2603.17558v1) | 共享A、B的共享/语言私有列及Static/Hard/Soft rank路由 | 12语言联合ASR长尾设置，额外language-ID/router；不是本轮单融合模型 | 语言条件共享/私有分配已有，语言迁移不能当新算法 |
| [Shared Logical Subspace ACL2026](https://aclanthology.org/2026.acl-long.1806.pdf) | NL/符号证明的残差池化，98% PCA→CCA→反投影QR，推理逐token放大投影 | 配对证明、固定k32；无参数更新/CL保护 | 多视图CCA与功能steering已有；这里系数梯度与残差激活不同但需实证额外信息 |
| [PRISM-Δ v1](https://arxiv.org/pdf/2603.10705v1) | ΩΔ=Hᵀ(H+−H−)/N的未中心化差分交叉矩SVD，分别steer K/V | 相关/无关问题三元组、推理额外计算，非多语言顺序更新 | 对照差分交叉协方差与SVD不是新数学；不能借其能量最优性推ACC |
| [Crosslingual Unlearning v1](https://arxiv.org/pdf/2601.06675v1) | UNLEARN对任务/语言表示分离作选择性干预分析 | 已读§4.2–4.3与原文方法概览；未审计全部实现；TOFU事实场景 | 作者报告的跨语言共享/语言残差因果实验已有；不泛化为本项目SFT证明 |
| [cPCA Nature Communications2018](https://doi.org/10.1038/s41467-018-04608-8) | 目标协方差减背景协方差的特征问题 | 一般统计方法，不是CL训练控制器 | 本文配对C=Cov((X+Y)/2)−Cov((X−Y)/2)的形式不新，识别语义仍待验证 |

新增来源哈希与读取范围写入来源清单；论文全文不重分发。不因清单更长就宣称检索穷尽。
