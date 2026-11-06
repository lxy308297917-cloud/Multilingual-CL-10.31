# 翻译解码开发对照

固定64个FLORES dev句组，两方向；Base与LoRA Final400；原greedy和官方non-thinking采样两套预定配置。排除原3个few-shot句组。仅为开发诊断，不覆盖正式表。

|模型|解码|方向|chrF++|512上限条数/64|
|---|---|---|---:|---:|
|base|greedy|en->sw|14.81|22|
|base|greedy|sw->en|35.82|0|
|base|official_non_thinking|en->sw|16.50|17|
|base|official_non_thinking|sw->en|35.52|0|
|lora|greedy|en->sw|7.55|44|
|lora|greedy|sw->en|36.52|0|
|lora|official_non_thinking|en->sw|13.06|26|
|lora|official_non_thinking|sw->en|35.53|0|

## 配对差异

|对比|模型或解码|方向|chrF++差异|配对95%区间|
|---|---|---|---:|---|
|official_sampling_minus_greedy|base|en->sw|+1.69|[-1.24, +4.36]|
|official_sampling_minus_greedy|lora|en->sw|+5.51|[+3.32, +8.25]|
|lora_minus_base|greedy|en->sw|-7.26|[-10.21, -4.57]|
|lora_minus_base|official_non_thinking|en->sw|-3.44|[-6.50, -0.38]|
|official_sampling_minus_greedy|base|sw->en|-0.30|[-1.21, +0.52]|
|official_sampling_minus_greedy|lora|sw->en|-0.99|[-2.86, +1.02]|
|lora_minus_base|greedy|sw->en|+0.70|[-1.84, +2.81]|
|lora_minus_base|official_non_thinking|sw->en|+0.01|[-2.53, +2.17]|

结论：采样减少EN→SW重复并提高当前开发集分数，但LoRA仍未超过相同采样下的Base。解码是可干预的影响因素，不能单凭此对照把原问题全部解释为greedy，也不能宣布翻译学习已经修复。SW→EN变化较小。

下一步先完成现有正式评测；有限数据修正需独立冻结开发规则。按任务匹配和多语言覆盖补充XNLI/XCOPA，不以补充成绩替代翻译收益。

区间为1000次配对bootstrap、seed42、未修正多重比较；只针对当前观察到的输出，不代表跨推理/训练种子稳定性。

官方采样参数来源：https://huggingface.co/Qwen/Qwen3-4B#best-practices 。
