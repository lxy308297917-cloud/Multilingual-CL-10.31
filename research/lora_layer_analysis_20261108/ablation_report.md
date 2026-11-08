# 实验A：训练后LoRA层级消融

已完成18/18组全量测评；尚未完成：无。

以下变化均相对同语言原Full LoRA；负数表示消融后下降。除摘要ROUGE-Lsum外单位为百分点。不以本表挑选下一轮训练层数。

|语言|消融|任务|Base|Full|消融|相对Full变化|配对95%区间|
|---|---|---|---:|---:|---:|---:|---|
|sw|remove_early|Belebele target ACC|27.33|38.11|42.11|+4.00|[+1.44,+6.78]|
|sw|remove_early|Belebele EN ACC|62.89|86.33|86.56|+0.22|[-1.33,+1.78]|
|sw|remove_early|GSM8K strict-match|61.03|59.74|54.13|-5.61|[-8.34,-2.73]|
|sw|remove_early|GSM8K flexible-extract|84.46|81.20|79.30|-1.90|[-3.94,+0.23]|
|sw|remove_early|IFEval prompt_level_strict_acc|78.37|49.35|56.01|+6.65|[+3.33,+10.17]|
|sw|remove_early|IFEval inst_level_strict_acc|84.65|62.47|67.63|+5.16|[+2.56,+8.08]|
|sw|remove_early|XL-Sum target ROUGE-Lsum|13.71|20.24|18.42|-1.82|[-2.53,-1.03]|
|sw|remove_early|Global-MMLU SW ACC|33.38|33.34|35.36|+2.02|[+1.32,+2.78]|
|sw|remove_middle|Belebele target ACC|27.33|38.11|37.00|-1.11|[-3.78,+1.67]|
|sw|remove_middle|Belebele EN ACC|62.89|86.33|82.22|-4.11|[-6.11,-2.22]|
|sw|remove_middle|GSM8K strict-match|61.03|59.74|64.52|+4.78|[+1.74,+7.96]|
|sw|remove_middle|GSM8K flexible-extract|84.46|81.20|84.91|+3.71|[+1.67,+5.69]|
|sw|remove_middle|IFEval prompt_level_strict_acc|78.37|49.35|78.19|+28.84|[+24.40,+33.09]|
|sw|remove_middle|IFEval inst_level_strict_acc|84.65|62.47|84.29|+21.82|[+18.53,+25.28]|
|sw|remove_middle|XL-Sum target ROUGE-Lsum|13.71|20.24|19.04|-1.20|[-1.92,-0.46]|
|sw|remove_middle|Global-MMLU SW ACC|33.38|33.34|35.37|+2.03|[+1.13,+2.88]|
|sw|remove_late|Belebele target ACC|27.33|38.11|38.22|+0.11|[-1.22,+1.44]|
|sw|remove_late|Belebele EN ACC|62.89|86.33|86.67|+0.33|[-0.11,+0.89]|
|sw|remove_late|GSM8K strict-match|61.03|59.74|59.59|-0.15|[-2.73,+2.35]|
|sw|remove_late|GSM8K flexible-extract|84.46|81.20|80.82|-0.38|[-2.05,+1.29]|
|sw|remove_late|IFEval prompt_level_strict_acc|78.37|49.35|52.68|+3.33|[+0.55,+6.28]|
|sw|remove_late|IFEval inst_level_strict_acc|84.65|62.47|64.63|+2.16|[-0.12,+4.23]|
|sw|remove_late|XL-Sum target ROUGE-Lsum|13.71|20.24|16.12|-4.11|[-4.96,-3.25]|
|sw|remove_late|Global-MMLU SW ACC|33.38|33.34|33.16|-0.18|[-0.58,+0.26]|
|sw|keep_late|Belebele target ACC|27.33|38.11|26.56|-11.56|[-14.78,-8.33]|
|sw|keep_late|Belebele EN ACC|62.89|86.33|71.56|-14.78|[-17.78,-12.00]|
|sw|keep_late|GSM8K strict-match|61.03|59.74|62.32|+2.58|[-0.53,+5.69]|
|sw|keep_late|GSM8K flexible-extract|84.46|81.20|83.55|+2.35|[+0.23,+4.55]|
|sw|keep_late|IFEval prompt_level_strict_acc|78.37|49.35|80.04|+30.68|[+26.25,+35.30]|
|sw|keep_late|IFEval inst_level_strict_acc|84.65|62.47|85.73|+23.26|[+19.83,+26.72]|
|sw|keep_late|XL-Sum target ROUGE-Lsum|13.71|20.24|14.08|-6.16|[-6.92,-5.41]|
|sw|keep_late|Global-MMLU SW ACC|33.38|33.34|33.77|+0.43|[-0.46,+1.35]|
|sw|remove_attention|Belebele target ACC|27.33|38.11|42.00|+3.89|[+1.56,+6.22]|
|sw|remove_attention|Belebele EN ACC|62.89|86.33|89.11|+2.78|[+1.33,+4.33]|
|sw|remove_attention|GSM8K strict-match|61.03|59.74|45.64|-14.10|[-17.06,-11.07]|
|sw|remove_attention|GSM8K flexible-extract|84.46|81.20|82.03|+0.83|[-1.06,+2.65]|
|sw|remove_attention|IFEval prompt_level_strict_acc|78.37|49.35|69.50|+20.15|[+16.27,+24.40]|
|sw|remove_attention|IFEval inst_level_strict_acc|84.65|62.47|77.94|+15.47|[+12.24,+18.61]|
|sw|remove_attention|XL-Sum target ROUGE-Lsum|13.71|20.24|19.93|-0.31|[-0.97,+0.39]|
|sw|remove_attention|Global-MMLU SW ACC|33.38|33.34|36.52|+3.18|[+2.44,+3.90]|
|sw|remove_mlp|Belebele target ACC|27.33|38.11|41.22|+3.11|[+0.22,+6.00]|
|sw|remove_mlp|Belebele EN ACC|62.89|86.33|87.89|+1.56|[-0.11,+3.22]|
|sw|remove_mlp|GSM8K strict-match|61.03|59.74|59.29|-0.45|[-3.64,+2.58]|
|sw|remove_mlp|GSM8K flexible-extract|84.46|81.20|85.52|+4.32|[+2.35,+6.37]|
|sw|remove_mlp|IFEval prompt_level_strict_acc|78.37|49.35|76.71|+27.36|[+22.74,+31.42]|
|sw|remove_mlp|IFEval inst_level_strict_acc|84.65|62.47|82.85|+20.38|[+17.07,+23.59]|
|sw|remove_mlp|XL-Sum target ROUGE-Lsum|13.71|20.24|15.36|-4.88|[-5.66,-4.10]|
|sw|remove_mlp|Global-MMLU SW ACC|33.38|33.34|36.28|+2.95|[+2.13,+3.77]|
|ar|remove_early|Belebele target ACC|46.22|78.44|79.00|+0.56|[-1.33,+2.33]|
|ar|remove_early|Belebele EN ACC|62.89|87.67|87.22|-0.44|[-1.78,+0.89]|
|ar|remove_early|GSM8K strict-match|61.03|40.03|35.25|-4.78|[-7.43,-2.05]|
|ar|remove_early|GSM8K flexible-extract|84.46|75.82|75.36|-0.45|[-2.50,+1.52]|
|ar|remove_early|IFEval prompt_level_strict_acc|78.37|42.70|47.50|+4.81|[+1.11,+8.32]|
|ar|remove_early|IFEval inst_level_strict_acc|84.65|56.47|60.55|+4.08|[+1.31,+6.91]|
|ar|remove_early|XL-Sum target ROUGE-Lsum|9.21|20.89|18.99|-1.90|[-2.19,-1.60]|
|ar|remove_middle|Belebele target ACC|46.22|78.44|79.56|+1.11|[-0.33,+2.78]|
|ar|remove_middle|Belebele EN ACC|62.89|87.67|85.78|-1.89|[-3.56,-0.22]|
|ar|remove_middle|GSM8K strict-match|61.03|40.03|63.00|+22.97|[+19.63,+26.16]|
|ar|remove_middle|GSM8K flexible-extract|84.46|75.82|85.82|+10.01|[+7.88,+12.13]|
|ar|remove_middle|IFEval prompt_level_strict_acc|78.37|42.70|76.16|+33.46|[+28.65,+38.08]|
|ar|remove_middle|IFEval inst_level_strict_acc|84.65|56.47|82.73|+26.26|[+22.79,+29.91]|
|ar|remove_middle|XL-Sum target ROUGE-Lsum|9.21|20.89|18.62|-2.27|[-2.56,-1.99]|
|ar|remove_late|Belebele target ACC|46.22|78.44|79.44|+1.00|[-0.33,+2.44]|
|ar|remove_late|Belebele EN ACC|62.89|87.67|88.22|+0.56|[+0.00,+1.11]|
|ar|remove_late|GSM8K strict-match|61.03|40.03|53.60|+13.57|[+11.14,+16.30]|
|ar|remove_late|GSM8K flexible-extract|84.46|75.82|78.39|+2.58|[+0.68,+4.47]|
|ar|remove_late|IFEval prompt_level_strict_acc|78.37|42.70|46.95|+4.25|[+1.11,+7.39]|
|ar|remove_late|IFEval inst_level_strict_acc|84.65|56.47|59.71|+3.24|[+0.95,+5.71]|
|ar|remove_late|XL-Sum target ROUGE-Lsum|9.21|20.89|19.93|-0.96|[-1.22,-0.71]|
|ar|keep_late|Belebele target ACC|46.22|78.44|50.44|-28.00|[-31.56,-24.11]|
|ar|keep_late|Belebele EN ACC|62.89|87.67|72.89|-14.78|[-17.33,-12.00]|
|ar|keep_late|GSM8K strict-match|61.03|40.03|63.38|+23.35|[+20.09,+26.54]|
|ar|keep_late|GSM8K flexible-extract|84.46|75.82|82.34|+6.52|[+4.09,+9.02]|
|ar|keep_late|IFEval prompt_level_strict_acc|78.37|42.70|80.41|+37.71|[+32.90,+42.33]|
|ar|keep_late|IFEval inst_level_strict_acc|84.65|56.47|86.09|+29.62|[+25.88,+33.21]|
|ar|keep_late|XL-Sum target ROUGE-Lsum|9.21|20.89|9.53|-11.36|[-11.67,-11.04]|
|ar|remove_attention|Belebele target ACC|46.22|78.44|81.78|+3.33|[+1.56,+5.11]|
|ar|remove_attention|Belebele EN ACC|62.89|87.67|89.00|+1.33|[+0.00,+2.67]|
|ar|remove_attention|GSM8K strict-match|61.03|40.03|51.48|+11.45|[+8.57,+14.40]|
|ar|remove_attention|GSM8K flexible-extract|84.46|75.82|81.20|+5.38|[+3.34,+7.43]|
|ar|remove_attention|IFEval prompt_level_strict_acc|78.37|42.70|59.33|+16.64|[+12.75,+20.70]|
|ar|remove_attention|IFEval inst_level_strict_acc|84.65|56.47|69.78|+13.31|[+10.25,+16.42]|
|ar|remove_attention|XL-Sum target ROUGE-Lsum|9.21|20.89|20.82|-0.08|[-0.32,+0.16]|
|ar|remove_mlp|Belebele target ACC|46.22|78.44|80.78|+2.33|[+0.44,+4.22]|
|ar|remove_mlp|Belebele EN ACC|62.89|87.67|88.44|+0.78|[-0.78,+2.22]|
|ar|remove_mlp|GSM8K strict-match|61.03|40.03|63.68|+23.65|[+20.47,+26.84]|
|ar|remove_mlp|GSM8K flexible-extract|84.46|75.82|83.02|+7.20|[+5.23,+9.33]|
|ar|remove_mlp|IFEval prompt_level_strict_acc|78.37|42.70|68.95|+26.25|[+22.00,+30.68]|
|ar|remove_mlp|IFEval inst_level_strict_acc|84.65|56.47|77.46|+20.98|[+17.65,+24.27]|
|ar|remove_mlp|XL-Sum target ROUGE-Lsum|9.21|20.89|17.24|-3.65|[-3.95,-3.34]|
|ru|remove_early|Belebele target ACC|55.56|81.89|81.22|-0.67|[-2.45,+1.11]|
|ru|remove_early|Belebele EN ACC|62.89|89.00|88.11|-0.89|[-2.33,+0.44]|
|ru|remove_early|GSM8K strict-match|61.03|21.08|27.22|+6.14|[+3.56,+8.72]|
|ru|remove_early|GSM8K flexible-extract|84.46|78.39|77.41|-0.99|[-2.96,+0.83]|
|ru|remove_early|IFEval prompt_level_strict_acc|78.37|42.33|45.47|+3.14|[-0.37,+6.47]|
|ru|remove_early|IFEval inst_level_strict_acc|84.65|56.35|58.63|+2.28|[-0.36,+4.97]|
|ru|remove_early|XL-Sum target ROUGE-Lsum|10.07|18.87|16.91|-1.97|[-2.18,-1.76]|
|ru|remove_middle|Belebele target ACC|55.56|81.89|84.67|+2.78|[+1.11,+4.44]|
|ru|remove_middle|Belebele EN ACC|62.89|89.00|88.56|-0.44|[-1.78,+0.89]|
|ru|remove_middle|GSM8K strict-match|61.03|21.08|52.92|+31.84|[+28.89,+35.03]|
|ru|remove_middle|GSM8K flexible-extract|84.46|78.39|85.22|+6.82|[+4.93,+8.79]|
|ru|remove_middle|IFEval prompt_level_strict_acc|78.37|42.33|77.08|+34.75|[+30.31,+39.00]|
|ru|remove_middle|IFEval inst_level_strict_acc|84.65|56.35|83.57|+27.22|[+23.70,+30.72]|
|ru|remove_middle|XL-Sum target ROUGE-Lsum|10.07|18.87|15.94|-2.94|[-3.16,-2.73]|
|ru|remove_late|Belebele target ACC|55.56|81.89|82.78|+0.89|[+0.33,+1.56]|
|ru|remove_late|Belebele EN ACC|62.89|89.00|88.67|-0.33|[-0.89,+0.22]|
|ru|remove_late|GSM8K strict-match|61.03|21.08|32.83|+11.75|[+9.55,+14.10]|
|ru|remove_late|GSM8K flexible-extract|84.46|78.39|79.98|+1.59|[-0.08,+3.26]|
|ru|remove_late|IFEval prompt_level_strict_acc|78.37|42.33|45.47|+3.14|[+0.00,+6.10]|
|ru|remove_late|IFEval inst_level_strict_acc|84.65|56.35|58.63|+2.28|[-0.12,+4.58]|
|ru|remove_late|XL-Sum target ROUGE-Lsum|10.07|18.87|18.15|-0.72|[-0.90,-0.54]|
|ru|keep_late|Belebele target ACC|55.56|81.89|56.00|-25.89|[-29.67,-22.44]|
|ru|keep_late|Belebele EN ACC|62.89|89.00|69.44|-19.56|[-22.44,-16.67]|
|ru|keep_late|GSM8K strict-match|61.03|21.08|63.91|+42.84|[+39.80,+45.94]|
|ru|keep_late|GSM8K flexible-extract|84.46|78.39|82.87|+4.47|[+2.20,+6.83]|
|ru|keep_late|IFEval prompt_level_strict_acc|78.37|42.33|78.56|+36.23|[+31.42,+40.85]|
|ru|keep_late|IFEval inst_level_strict_acc|84.65|56.35|84.89|+28.54|[+24.82,+32.15]|
|ru|keep_late|XL-Sum target ROUGE-Lsum|10.07|18.87|10.43|-8.44|[-8.68,-8.22]|
|ru|remove_attention|Belebele target ACC|55.56|81.89|84.00|+2.11|[+0.67,+3.67]|
|ru|remove_attention|Belebele EN ACC|62.89|89.00|89.33|+0.33|[-0.89,+1.56]|
|ru|remove_attention|GSM8K strict-match|61.03|21.08|18.35|-2.73|[-5.08,-0.38]|
|ru|remove_attention|GSM8K flexible-extract|84.46|78.39|81.12|+2.73|[+0.91,+4.47]|
|ru|remove_attention|IFEval prompt_level_strict_acc|78.37|42.33|60.26|+17.93|[+14.05,+22.00]|
|ru|remove_attention|IFEval inst_level_strict_acc|84.65|56.35|70.38|+14.03|[+11.07,+17.10]|
|ru|remove_attention|XL-Sum target ROUGE-Lsum|10.07|18.87|18.42|-0.45|[-0.63,-0.27]|
|ru|remove_mlp|Belebele target ACC|55.56|81.89|83.33|+1.44|[-0.11,+3.11]|
|ru|remove_mlp|Belebele EN ACC|62.89|89.00|88.56|-0.44|[-1.78,+0.89]|
|ru|remove_mlp|GSM8K strict-match|61.03|21.08|49.05|+27.98|[+25.09,+30.86]|
|ru|remove_mlp|GSM8K flexible-extract|84.46|78.39|82.26|+3.87|[+1.90,+5.91]|
|ru|remove_mlp|IFEval prompt_level_strict_acc|78.37|42.33|63.96|+21.63|[+17.38,+25.69]|
|ru|remove_mlp|IFEval inst_level_strict_acc|84.65|56.35|73.26|+16.91|[+13.57,+20.28]|
|ru|remove_mlp|XL-Sum target ROUGE-Lsum|10.07|18.87|15.99|-2.88|[-3.10,-2.66]|

## 解释边界
只用seed42，区间未做多重校正；IFEval指令级按提示cluster抽样。保持所有六组，不基于正式测试选k；关闭已训练增量不是部分层重新训练，层间不匹配会影响消融。范数不等于功能重要性，观察关联不证明子空间干扰因果。实验B/C/D尚未运行。
