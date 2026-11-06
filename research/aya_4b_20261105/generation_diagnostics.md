# 生成长度与截断诊断

诊断不参与方法选择或调参；触及长度上限可能同时反映长答案、重复输出或截断，不能单独认定原因。所有模型使用冻结greedy协议，保留原始输出供本地复核。

|模型|任务|样本|触及生成长度上限|空输出|输入文章截断|
|---|---|---:|---:|---:|---:|
|Base|flores200_en_sw_full_devtest_v2|1012|345|0|0|
|Base|flores200_sw_en_full_devtest_v2|1012|0|0|0|
|Base|xlsum_sw_v2|987|976|0|0|
|Base|xlsum_en_v2|11535|329|0|0|
|FFT|flores200_en_sw_full_devtest_v2|1012|811|0|0|
|FFT|flores200_sw_en_full_devtest_v2|1012|8|0|0|
|FFT|xlsum_sw_v2|987|189|0|0|
|FFT|xlsum_en_v2|11535|1520|0|0|
|LoRA|flores200_en_sw_full_devtest_v2|1012|769|0|0|
|LoRA|flores200_sw_en_full_devtest_v2|1012|5|0|0|
|LoRA|xlsum_sw_v2|987|137|0|0|
|LoRA|xlsum_en_v2|11535|101|0|0|
|SSU-Wanda|flores200_en_sw_full_devtest_v2|1012|668|0|0|
|SSU-Wanda|flores200_sw_en_full_devtest_v2|1012|11|0|0|
|SSU-Wanda|xlsum_sw_v2|987|129|0|0|
|SSU-Wanda|xlsum_en_v2|11535|2351|0|0|
