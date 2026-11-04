> 这是首批开始前冻结的协议存档；WikiLingua原404状态已由[恢复补充协议](wikilingua_protocol_addendum.md)更新。其他生成/评分设置保持不变。

# 冻结评测协议：统一语言顺序基线

本轮独立于Aya SW→HA。原模型为官方后训练`Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，不是另一个Qwen3-Base。

## 比较结构与用途

每轨独立比较Base、Single ES Final、Single TH Final、同一个Single ES Final→TH Final。第一轮不训练FR，不反转顺序。正式主模型固定预算Final；中间checkpoint100只用于学习曲线，不用于挑模型。

训练每段3200不同源ID、200 optimizer steps、有效batch16、seed42、AdamW LR1e-5/weight_decay0、cosine/warmup5%、BF16、assistant-only、non-thinking、cutoff2048、不packing。SIB-200独立采用640条/40步。第二阶段只加载ES最终权重，重新初始化优化器与scheduler。实际输入/移位回答监督token、重复内容、显存和GPU成本分别记录；不同语言/任务不宣称同token或同计算量。

## 固定任务读出

| 轨道 | 开发材料 | 正式范围/语言 | 首要指标 | 补充与格式诊断 |
|---|---|---|---|---|
| MASSIVE | 官方dev按共同源ID SHA固定256题/语言 | 官方test，ES/TH/FR/EN各2974 | 生成intent标签ACC | 非法标签率、观察到的参考类别macro accuracy/F1；固定dev64全标签sum-log-prob单列 |
| MathOctopus | 独立训练源256共同题ID，不进入训练 | MGSM-Rev2各250、MSVAMP各1000，ES/TH/FR/EN分别报告 | 最终数值ACC（固定宽松数字提取） | 严格`####`格式ACC、解析率、生成长度、截断；不把纯格式改善当数学净收益 |
| XNLI | 官方validation固定共同256注释ID/语言 | 官方test实际各5010题，四语言 | 生成固定三分类标签ACC | 非法标签率、macro-F1、固定dev64候选概率 |
| MTOP | 官方eval固定共同256官方ID/语言 | 官方语言原生test；ES2998、TH2765、FR3193、EN4386 | 完整语义树结构EM | intent ACC、合法树率。叶子token顺序/大小写保留，忽略括号空白；不把无效输出全当作语义错误的唯一解释 |
| WikiLingua | NOT_RUN | NOT_RUN | 计划ROUGE-Lsum | 作者防泄漏split入口404；不能用HF仅有的train擅造官方test |
| XL-Sum | 各语言官方dev独立SHA选256篇 | 官方test ES4763、TH826、FR1086、EN11535 | 作者多语言ROUGE-Lsum F1 | ROUGE-1/2、输出/参考长度、自动语言诊断和正文覆盖率；非平行新闻，不做逐语言题目配对 |
| SIB-200 | 官方validation全部99题/语言 | 官方test各204，四语言 | 生成7类主题标签ACC | Macro-F1、非法标签率，固定dev64候选概率 |

分类只去首尾空白，要求完整输出等于合法标签；markdown、解释和别名属于非法输出。候选评分不是另一套替换主成绩的评分：仅在提前固定的64条开发题上，所有合法标签的目标token log-prob求和argmax；不添加EOS，不对分数按长度归一化。按同样64条ID对照生成结果，报告格式与判别差异。

数学数字提取规则固定在`common.py`：最后数值canonical Decimal，处理英文千位逗号与ES/FR十进制逗号，无法解析计错；严格格式另用`####`。不执行输出中的任何代码。歧义数字格式及最后数字策略可能误判，保留输出和人工错误分析，不按模型表现更换解析器。

MTOP使用官方第7列compositional decoupled表示，不把flat intent或slot列表当完整结构。v1开发提示没明确方括号语法，出现XML/重复标签；工程验真后一次性加入语法说明，另建`prepared-v2`并重测Base开发，v1结果保留且不混用。没有使用正式test选择提示。固定Final仍报告格式地板，不承诺一定学会任务。

## 提示与生成

统一使用官方LLaMA-Factory `qwen3`、`enable_thinking=False`编码的source token IDs，目标语言任务说明、固定英文标签或官方树标签。源token逐条保留并按哈希验证，不另换eval聊天模板。0-shot，无测试示例。vLLM BF16、greedy/temperature0、seed42、TP1；分类max_new24，MTOP256，数学1280，摘要512，所有对照同一设置。数学/分类/结构max_model_len8192，摘要4096；不按模型表现调整长度。

XL-Sum原始长文本审计发现TH 131/256、ES57/256的文章+摘要超过2048。因此维持总训练长度2048，提前冻结1536 source（含任务说明及完整assistant头）和最多512训练target：正文取能容纳的最长token前缀，完整正式参考不截断。该策略在训练、开发、正式评测及所有模型相同，记录每条正文覆盖率。长target训练监督前缀另记，不能声称完整文章/参考都被用于训练。

摘要评分使用XL-Sum作者`multilingual_rouge_scoring@afcd803c8d7c98f3be60e9a7ce57fcbc3e729e8c`，ES/FR/EN Snowball stemming，TH PyThaiNLP5.1.2 `newmm`且不做词干化；pyonmttok1.37.1。相同文本得100/空输出得0的四语言独立例子已验真。按原生newline计算ROUGE-Lsum，不把默认仅拉丁文字的regex当泰语分词；自动langdetect仅辅助，不等于人工语言认证。

## 旧能力监测

Base与每个阶段Final统一记录：项目原协议EN Belebele900 0-shot ACC、EN GSM8K1319 5-shot strict/flexible、官方IFEval541提示/834指令strict（loose单列）。旧任务定义、源数据和原始逐题输出保留，不按EN下降提前停训，不把几个英语分数合并成一个遗忘数。

## 效应与统计

分别报告`G_ES=S_ES(W1)-S_ES(W0)`、`G_TH=S_TH(W2)-S_TH(W1)`、`F_ES=S_ES(W1)-S_ES(W2)`、`T_TH=S_TH(W1)-S_TH(W0)`。Single TH还要报告相对Base的TH收益，才能比较第二阶段与单语言学习；不要只比较最终绝对分数。正值F_ES表示下降，负值表示进一步提高，不人为制造遗忘。

同一语言、同一任务、同一题在模型间配对bootstrap；平行语言的联合分析按可靠源ID成组；Belebele按文章cluster，IFEval按提示为抽样单位并保留各提示指令数。XL-Sum各语言独立，模型之间仍可对同一语言文章配对。固定模型的题目区间不是训练种子方差。首轮只有seed42，不追加多seed；未做多重比较校正，区间跨零不能证明等效。

正式测试只报告与解释，不挑checkpoint、学习率、语言顺序或停止节点。已暴露开发/历史相关测试如实记录；预训练污染不能由本次必要精确检查排除。未运行项写NOT_RUN，不将训练loss/NLL或格式通过率单独当目标任务收益。
