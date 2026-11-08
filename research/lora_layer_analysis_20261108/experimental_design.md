# 实验A：设计、模型与解释范围

## 研究问题

本轮观察既有多语言SFT增量在深度和模块类型上的分布，以及关闭部分增量之后的任务取舍。研究基线为`d92c63b`；三种语言分别使用既有固定Final400 LoRA。对每种语言完成相同的六组训练后消融，正式测试不用于选择下一轮训练的层数、超参数或方法。

本轮新增训练为0。Full LoRA表示全部36层的Adapter启用；Base表示适配前的官方`Qwen/Qwen3-4B`，revision为`1cfa9a7208912126459214e8b04321603b3df60c`。模型名称中的Base角色与官方模型仓库名需区分，本实验使用的官方仓库没有`-Base`后缀。

## 来源模型与训练曝光

三种来源模型均完成6400条样本、400个optimizer steps、1 epoch；seed和data_seed均为42。LoRA rank16、alpha32、dropout0.05，覆盖每层q/k/v/o/gate/up/down七个投影。AdamW学习率1e-4、cosine、warmup0.05；micro-batch1、梯度累积16，有效batch16；BF16、长度2048、无packing、assistant-only、non-thinking。模型固定Final，没有按英语保持门槛选模。

|语言|训练样本|输入token|移位后的assistant监督token|
|---|---:|---:|---:|
|SW|6400|2,373,891|324,892|
|AR|6400|2,265,373|265,783|
|RU|6400|2,520,204|305,046|

因此三语言对齐了样本与步数，没有对齐有效监督token。它们是分别从同一官方模型适配的三份模型，不是SW→AR→RU顺序训练。SW使用原4B混合数据对应的LoRA，不使用后来清洗翻译或NLLB替换版本。来源配置和权重指纹见`source_training_configuration.json`、`adapter_ablation_audit.json`。

## A1：有效更新范数

对每个投影计算有效更新`ΔW=(alpha/r)BA=2BA`的Frobenius范数，并计算`||ΔW||F/||W||F`。Gram恒等式在FP64中计算更新范数，与显式矩阵乘积的独立小矩阵检查一致；主干权重范数在FP32中计算。单独的A或B范数不作为更新幅度。

共有3语言×36层×7投影=756条模块记录。分组汇总使用平方范数相加，再开方得到拼接更新的范数；Attention/MLP占比按平方范数报告。幅度及相对幅度都不等同于功能重要性，跨语言层幅度相关也不等于更新方向相同。

## A2：六组训练后消融

36层预先均分为前部0–11、中部12–23、后部24–35。每份来源模型独立复制，禁用模块只将LoRA B置零；保留模块的全部张量与来源逐张量一致，来源权重保持不变。

|消融|仍启用的模块数|仍启用的Adapter参数|禁用的Adapter参数|
|---|---:|---:|---:|
|Remove-Early|168|22,020,096|11,010,048|
|Remove-Middle|168|22,020,096|11,010,048|
|Remove-Late|168|22,020,096|11,010,048|
|Keep-Late|84|11,010,048|22,020,096|
|Remove-Attention（保留MLP）|108|21,233,664|11,796,480|
|Remove-MLP（保留Attention）|144|11,796,480|21,233,664|

Full来源有252模块、33,030,144个Adapter参数。上表描述推理时仍启用增量的范围；各行没有重新训练，零化B也没有减少主干的推理FLOPs。模块组和深度组的更新量、模块数及更新范数不完全匹配，这些差异会影响机制解释。

## A3：冻结测评与统计

|任务|每模型样本|主要指标|固定协议|
|---|---:|---|---|
|目标语言/英语Belebele|各900|ACC|0-shot，HF四候选字母似然，batch4|
|SW Global-MMLU|14,042，57学科|逐题加权ACC|5-shot，HF，batch1|
|目标XL-Sum|SW987；AR4689；RU7780|ROUGE-Lsum|0-shot，greedy，最多256新token，batch8|
|英语GSM8K|1319|官方flexible-extract ACC|5-shot，greedy，最多1280新token|
|英语IFEval|541 prompts/834 instructions|官方prompt strict|0-shot，greedy，最多1280新token|

GSM8K strict-match和IFEval instruction strict同步完整保留。XL-Sum其余ROUGE和chrF++作为摘要辅助指标保留；摘要chrF++不属于翻译测评。SW摘要沿用原固定评分，AR/RU沿用原生Unicode分词与显式sentence-LCS；不跨语言比较ROUGE绝对高低。

Base/Full直接复用相同冻结协议的既有结果。消融与Full逐题匹配doc/prompt/target指纹，阅读重新从原始候选似然argmax核对ACC。每项变化相对Full进行2000次配对bootstrap；IFEval指令级按prompt cluster抽样，避免把同一prompt的指令当作独立题。区间没有多重比较校正，仅反映当前seed42模型和题目条件，不提供跨训练种子的稳定性证据。

这些正式题目此前已暴露，本轮属于预先定义掩码的补充比较。没有根据结果修改提示、解码、评分、样本量或checkpoint。阅读的选项位置偏好、摘要分词和最终答案格式均需要在解释中考虑。

## 因果与后续实验的边界

训练后删除更新可能产生层间不匹配；观察到某组恢复通用分数，仍需同步检查目标任务损失。没有等范数或匹配随机消融，不能仅凭这些结果宣称定位了独立的语言或通用能力回路。

Keep-Late只检验现有Full增量的子集，不代表只训练末12层会得到相同模型。LinguaMap使用不同规模模型与真实选择性SFT；本轮是受其启发的LoRA消融，具体差异见`paper_scope.md`。

实验B/C/D尚未运行。若审查A后开展层选择训练，可沿用预定义k=12比较First/Middle/Last/Random/All；Random在训练前一次性冻结，各组参数预算和实际曝光量分别报告。正式测试不用于挑选k或训练配置。新翻译、单层消融、隐藏状态、Logit Lens、梯度诊断及等范数随机控制均未在本轮运行。
