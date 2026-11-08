# 探索性：Early/Middle在Late启用条件下的分数交互

此分析在观察SW首批阅读及核心结果后提出，使用原预设四个掩码，不新增推理。不是预注册主结果，不用于选择B的层数或新方法。

定义：I = S(Full) − S(Remove-Early) − S(Remove-Middle) + S(Keep-Late)。它只描述这个训练后Adapter和指标上的非加性；ACC阈值、指标非线性及层间不匹配都可能产生交互，不能据此宣称梯度冲突或神经元功能定位。

|语言|任务|交互|配对95%区间|
|---|---|---:|---|
|sw|Belebele target ACC|-14.44|[-18.67,-10.56]|
|sw|Belebele EN ACC|-10.89|[-13.44,-8.33]|
|sw|GSM8K strict-match|+3.41|[-0.99,+7.58]|
|sw|GSM8K flexible-extract|+0.53|[-2.43,+3.41]|
|sw|IFEval prompt_level_strict_acc|-4.81|[-9.43,-0.55]|
|sw|IFEval inst_level_strict_acc|-3.72|[-7.29,-0.47]|
|sw|XL-Sum target ROUGE-Lsum|-3.13|[-4.23,-2.15]|
|ar|Belebele target ACC|-29.67|[-33.56,-26.00]|
|ar|Belebele EN ACC|-12.44|[-15.22,-9.67]|
|ar|GSM8K strict-match|+5.16|[+0.68,+9.40]|
|ar|GSM8K flexible-extract|-3.03|[-5.84,-0.23]|
|ar|IFEval prompt_level_strict_acc|-0.55|[-5.73,+4.25]|
|ar|IFEval inst_level_strict_acc|-0.72|[-4.56,+2.80]|
|ar|XL-Sum target ROUGE-Lsum|-7.19|[-7.56,-6.81]|
|ru|Belebele target ACC|-28.00|[-31.67,-24.33]|
|ru|Belebele EN ACC|-18.22|[-21.11,-15.22]|
|ru|GSM8K strict-match|+4.85|[+0.45,+9.55]|
|ru|GSM8K flexible-extract|-1.36|[-3.94,+1.36]|
|ru|IFEval prompt_level_strict_acc|-1.66|[-6.47,+2.96]|
|ru|IFEval inst_level_strict_acc|-0.96|[-4.36,+2.69]|
|ru|XL-Sum target ROUGE-Lsum|-3.54|[-3.80,-3.29]|

除摘要为ROUGE分外，交互单位均为百分点。单seed42，2000次同题配对bootstrap；指令级按prompt cluster。未做多重比较校正，不外推为跨种子稳定或跨模型机制。Global-MMLU不在此额外探索范围，原完整分数仍保留在主结果表。
