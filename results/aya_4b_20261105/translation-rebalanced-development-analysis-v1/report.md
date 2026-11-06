# 翻译解码开发对照

固定64个FLORES dev句组，两方向；Base与LoRA Final400；原greedy和官方non-thinking采样两套预定配置。排除原3个few-shot句组。仅为开发诊断，不覆盖正式表。

|模型|解码|方向|chrF++|512上限条数/64|
|---|---|---|---:|---:|
|base|greedy|en->sw|14.27|25|
|base|greedy|sw->en|35.94|0|
|base|official_non_thinking|en->sw|16.82|16|
|base|official_non_thinking|sw->en|35.58|0|
|lora|greedy|en->sw|10.17|40|
|lora|greedy|sw->en|37.46|0|
|lora|official_non_thinking|en->sw|17.95|12|
|lora|official_non_thinking|sw->en|36.05|0|

## 配对差异

|对比|模型或解码|方向|chrF++差异|配对95%区间|
|---|---|---|---:|---|
|official_sampling_minus_greedy|base|en->sw|+2.55|[-0.02, +5.40]|
|official_sampling_minus_greedy|lora|en->sw|+7.77|[+5.05, +10.66]|
|lora_minus_base|greedy|en->sw|-4.10|[-6.87, -1.63]|
|lora_minus_base|official_non_thinking|en->sw|+1.13|[-2.79, +4.90]|
|official_sampling_minus_greedy|base|sw->en|-0.35|[-1.23, +0.42]|
|official_sampling_minus_greedy|lora|sw->en|-1.41|[-3.71, +0.80]|
|lora_minus_base|greedy|sw->en|+1.52|[-1.04, +3.80]|
|lora_minus_base|official_non_thinking|sw->en|+0.47|[-1.93, +2.74]|

## 当前权重与Base的开发差异（官方non-thinking采样）
- en->sw：chrF++差异+1.13，配对95%区间[-2.79, +4.90]。
- sw->en：chrF++差异+0.47，配对95%区间[-1.93, +2.74]。

重复、长度上限与分数变化按各组真实统计解释。此处只验证当前固定权重和预定义解码；不把采样改善直接解释为训练学会了翻译，也不覆盖正式表。

有限数据修正只依据冻结开发规则推进，正式test不挑配置。

区间为1000次配对bootstrap、seed42、未修正多重比较；只针对当前观察到的输出，不代表跨推理/训练种子稳定性。

官方采样参数来源：https://huggingface.co/Qwen/Qwen3-4B#best-practices 。
