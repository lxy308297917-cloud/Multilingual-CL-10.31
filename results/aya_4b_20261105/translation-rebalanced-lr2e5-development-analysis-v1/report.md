# 翻译解码开发对照

固定64个FLORES dev句组，两方向；Base与LoRA Final400；原greedy和官方non-thinking采样两套预定配置。排除原3个few-shot句组。仅为开发诊断，不覆盖正式表。

|模型|解码|方向|chrF++|512上限条数/64|
|---|---|---|---:|---:|
|base|greedy|en->sw|14.16|25|
|base|greedy|sw->en|35.84|0|
|base|official_non_thinking|en->sw|16.13|18|
|base|official_non_thinking|sw->en|35.65|0|
|lora|greedy|en->sw|8.22|45|
|lora|greedy|sw->en|35.39|0|
|lora|official_non_thinking|en->sw|13.85|24|
|lora|official_non_thinking|sw->en|35.34|0|

## 配对差异

|对比|模型或解码|方向|chrF++差异|配对95%区间|
|---|---|---|---:|---|
|official_sampling_minus_greedy|base|en->sw|+1.98|[-0.58, +4.59]|
|official_sampling_minus_greedy|lora|en->sw|+5.63|[+3.56, +7.80]|
|lora_minus_base|greedy|en->sw|-5.94|[-8.38, -3.78]|
|lora_minus_base|official_non_thinking|en->sw|-2.29|[-5.17, +0.22]|
|official_sampling_minus_greedy|base|sw->en|-0.19|[-1.08, +0.58]|
|official_sampling_minus_greedy|lora|sw->en|-0.05|[-1.93, +1.83]|
|lora_minus_base|greedy|sw->en|-0.45|[-2.51, +1.68]|
|lora_minus_base|official_non_thinking|sw->en|-0.31|[-2.58, +1.77]|

## 当前权重与Base的开发差异（官方non-thinking采样）
- en->sw：chrF++差异-2.29，配对95%区间[-5.17, +0.22]。
- sw->en：chrF++差异-0.31，配对95%区间[-2.58, +1.77]。

重复、长度上限与分数变化按各组真实统计解释。此处只验证当前固定权重和预定义解码；不把采样改善直接解释为训练学会了翻译，也不覆盖正式表。

有限数据修正只依据冻结开发规则推进，正式test不挑配置。

区间为1000次配对bootstrap、seed42、未修正多重比较；只针对当前观察到的输出，不代表跨推理/训练种子稳定性。

官方采样参数来源：https://huggingface.co/Qwen/Qwen3-4B#best-practices 。
