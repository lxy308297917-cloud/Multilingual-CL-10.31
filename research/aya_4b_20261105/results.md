# 固定Final正式结果

生成测评统一使用内存配置v2：MT/摘要batch8、最多8序列、4096预填充token、显存比例0.65；GSM8K/IFEval最多8序列、8192批量token、显存比例0.65。所有模型沿用相同题目、提示、解码长度和评分；v1尝试保留，未混入本表。

ACC、IFEval为百分比；摘要与翻译为指标分，所有指标越高越好。

|任务／指标|Base|FFT|LoRA|SSU-Wanda|
|---|---:|---:|---:|---:|
|SW阅读 ACC|27.33|39.78|38.11|44.11|
|SW知识 ACC|33.38|32.72|33.34|34.67|
|SW摘要 ROUGE-Lsum|13.71|18.36|20.24|19.89|
|EN→SW chrF++|14.90|7.08|8.04|8.71|
|SW→EN chrF++|36.07|29.02|34.74|32.15|
|EN阅读 ACC|62.89|87.00|86.33|89.22|
|EN知识 ACC|66.74|68.09|67.92|69.33|
|EN摘要 ROUGE-Lsum|12.96|17.84|23.54|14.72|
|GSM8K flexible ACC|84.46|81.88|81.20|81.58|
|GSM8K strict ACC|61.03|47.92|59.74|42.08|
|IFEval prompt strict|78.37|44.92|49.35|69.69|
|IFEval instruction strict|84.65|56.71|62.47|79.02|

## 相对Base变化

ACC/IFEval变化单位为百分点，其余为指标分。区间未经多重比较校正，仅反映当前seed42模型的样本不确定性。Belebele按共享段落聚类，IFEval指令指标按prompt聚类；翻译重采样后重算语料级分数。

|任务／指标|方法|相对Base变化|配对95%区间|
|---|---|---:|---|
|SW阅读 ACC|FFT|+12.44|[+8.45, +16.04]|
|SW阅读 ACC|LoRA|+10.78|[+7.58, +14.27]|
|SW阅读 ACC|SSU-Wanda|+16.78|[+12.58, +20.71]|
|SW知识 ACC|FFT|-0.66|[-1.61, +0.33]|
|SW知识 ACC|LoRA|-0.04|[-0.94, +0.90]|
|SW知识 ACC|SSU-Wanda|+1.30|[+0.44, +2.20]|
|SW摘要 ROUGE-Lsum|FFT|+4.65|[+3.96, +5.44]|
|SW摘要 ROUGE-Lsum|LoRA|+6.53|[+5.74, +7.24]|
|SW摘要 ROUGE-Lsum|SSU-Wanda|+6.18|[+5.42, +6.92]|
|EN→SW chrF++|FFT|-7.82|[-8.43, -7.21]|
|EN→SW chrF++|LoRA|-6.86|[-7.48, -6.22]|
|EN→SW chrF++|SSU-Wanda|-6.19|[-6.83, -5.57]|
|SW→EN chrF++|FFT|-7.05|[-7.90, -6.23]|
|SW→EN chrF++|LoRA|-1.33|[-2.12, -0.62]|
|SW→EN chrF++|SSU-Wanda|-3.92|[-4.67, -3.12]|
|EN阅读 ACC|FFT|+24.11|[+21.03, +27.43]|
|EN阅读 ACC|LoRA|+23.44|[+20.32, +26.79]|
|EN阅读 ACC|SSU-Wanda|+26.33|[+23.49, +29.54]|
|EN知识 ACC|FFT|+1.35|[+0.66, +2.02]|
|EN知识 ACC|LoRA|+1.18|[+0.58, +1.78]|
|EN知识 ACC|SSU-Wanda|+2.59|[+2.02, +3.13]|
|EN摘要 ROUGE-Lsum|FFT|+4.88|[+4.72, +5.05]|
|EN摘要 ROUGE-Lsum|LoRA|+10.58|[+10.38, +10.78]|
|EN摘要 ROUGE-Lsum|SSU-Wanda|+1.76|[+1.65, +1.87]|
|GSM8K flexible ACC|FFT|-2.58|[-4.85, -0.45]|
|GSM8K flexible ACC|LoRA|-3.26|[-5.53, -1.14]|
|GSM8K flexible ACC|SSU-Wanda|-2.88|[-5.08, -0.83]|
|GSM8K strict ACC|FFT|-13.12|[-16.45, -9.86]|
|GSM8K strict ACC|LoRA|-1.29|[-4.55, +2.05]|
|GSM8K strict ACC|SSU-Wanda|-18.95|[-22.29, -15.69]|
|IFEval prompt strict|FFT|-33.46|[-38.82, -28.83]|
|IFEval prompt strict|LoRA|-29.02|[-34.20, -24.21]|
|IFEval prompt strict|SSU-Wanda|-8.69|[-12.39, -4.99]|
|IFEval instruction strict|FFT|-27.94|[-31.96, -24.17]|
|IFEval instruction strict|LoRA|-22.18|[-26.05, -18.70]|
|IFEval instruction strict|SSU-Wanda|-5.64|[-8.26, -3.03]|
