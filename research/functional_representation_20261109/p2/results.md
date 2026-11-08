# P2全部正式端点结果

所有新模型均为同目标6400条、400步Final。三项新对照共同batch2；旧batch1 Full只作历史旁证。除ROUGE-Lsum外均为百分比，越高越好。

|模型|SW阅读ACC|SW摘要ROUGE-Lsum|EN阅读ACC|SW Global-MMLU ACC|GSM8K flexible|IFEval prompt strict|
|---|---:|---:|---:|---:|---:|---:|
|适配前模型|27.33|13.71|62.89|33.38|84.46|78.37|
|旧Full LoRA（batch1，历史）|38.11|20.24|86.33|33.34|81.20|49.35|
|新All-layer LoRA（batch2）|36.33|20.18|87.56|33.03|81.05|48.43|
|普通全表示保持|40.89|20.58|88.44|35.34|84.76|79.85|
|同池英语CE回放|37.00|20.84|87.22|33.75|62.47|59.52|

## 严格提取与两级指令指标

|模型|GSM8K strict|GSM8K flexible|IFEval prompt strict|IFEval instruction strict|
|---|---:|---:|---:|---:|
|适配前模型|61.03|84.46|78.37|84.65|
|旧Full LoRA（batch1，历史）|59.74|81.20|49.35|62.47|
|新All-layer LoRA（batch2）|63.76|81.05|48.43|61.27|
|普通全表示保持|75.59|84.76|79.85|86.09|
|同池英语CE回放|61.03|62.47|59.52|69.78|

## 48项配对变化

差值为模型减参考；阅读、知识、数学、指令为百分点，摘要为ROUGE分。2000次配对bootstrap；指令级按提示聚类。单训练seed，未多重校正，未设非劣效容忍度；区间跨零不证明等效。

|模型|参考|任务／指标|变化|配对95%区间|
|---|---|---|---:|---|
|新All-layer LoRA（batch2）|适配前模型|Belebele target／ACC|+9.00|[+5.78, +12.22]|
|普通全表示保持|适配前模型|Belebele target／ACC|+13.56|[+10.00, +16.89]|
|普通全表示保持|新All-layer LoRA（batch2）|Belebele target／ACC|+4.56|[+1.89, +7.11]|
|普通全表示保持|同池英语CE回放|Belebele target／ACC|+3.89|[+1.56, +6.22]|
|同池英语CE回放|适配前模型|Belebele target／ACC|+9.67|[+6.56, +12.89]|
|同池英语CE回放|新All-layer LoRA（batch2）|Belebele target／ACC|+0.67|[-1.33, +2.78]|
|新All-layer LoRA（batch2）|适配前模型|Belebele EN／ACC|+24.67|[+21.56, +27.56]|
|普通全表示保持|适配前模型|Belebele EN／ACC|+25.56|[+22.55, +28.56]|
|普通全表示保持|新All-layer LoRA（batch2）|Belebele EN／ACC|+0.89|[-0.33, +2.22]|
|普通全表示保持|同池英语CE回放|Belebele EN／ACC|+1.22|[-0.22, +2.67]|
|同池英语CE回放|适配前模型|Belebele EN／ACC|+24.33|[+21.33, +27.44]|
|同池英语CE回放|新All-layer LoRA（batch2）|Belebele EN／ACC|-0.33|[-1.67, +0.89]|
|新All-layer LoRA（batch2）|适配前模型|XL-Sum target／ROUGE-Lsum|+6.47|[+5.73, +7.23]|
|普通全表示保持|适配前模型|XL-Sum target／ROUGE-Lsum|+6.87|[+6.14, +7.65]|
|普通全表示保持|新All-layer LoRA（batch2）|XL-Sum target／ROUGE-Lsum|+0.40|[-0.10, +0.93]|
|普通全表示保持|同池英语CE回放|XL-Sum target／ROUGE-Lsum|-0.25|[-0.79, +0.28]|
|同池英语CE回放|适配前模型|XL-Sum target／ROUGE-Lsum|+7.13|[+6.37, +7.85]|
|同池英语CE回放|新All-layer LoRA（batch2）|XL-Sum target／ROUGE-Lsum|+0.66|[+0.06, +1.24]|
|新All-layer LoRA（batch2）|适配前模型|GSM8K／strict-match|+2.73|[-0.53, +5.76]|
|普通全表示保持|适配前模型|GSM8K／strict-match|+14.56|[+12.21, +16.83]|
|普通全表示保持|新All-layer LoRA（batch2）|GSM8K／strict-match|+11.83|[+8.87, +14.71]|
|普通全表示保持|同池英语CE回放|GSM8K／strict-match|+14.56|[+11.60, +17.59]|
|同池英语CE回放|适配前模型|GSM8K／strict-match|+0.00|[-3.11, +2.96]|
|同池英语CE回放|新All-layer LoRA（batch2）|GSM8K／strict-match|-2.73|[-5.91, +0.45]|
|新All-layer LoRA（batch2）|适配前模型|GSM8K／flexible-extract|-3.41|[-5.61, -1.21]|
|普通全表示保持|适配前模型|GSM8K／flexible-extract|+0.30|[-1.36, +2.12]|
|普通全表示保持|新All-layer LoRA（batch2）|GSM8K／flexible-extract|+3.71|[+1.52, +5.91]|
|普通全表示保持|同池英语CE回放|GSM8K／flexible-extract|+22.29|[+19.41, +25.17]|
|同池英语CE回放|适配前模型|GSM8K／flexible-extract|-21.99|[-24.79, -19.11]|
|同池英语CE回放|新All-layer LoRA（batch2）|GSM8K／flexible-extract|-18.57|[-21.23, -16.00]|
|新All-layer LoRA（batch2）|适配前模型|IFEval／prompt_level_strict_acc|-29.94|[-34.57, -25.14]|
|普通全表示保持|适配前模型|IFEval／prompt_level_strict_acc|+1.48|[-1.66, +4.44]|
|普通全表示保持|新All-layer LoRA（batch2）|IFEval／prompt_level_strict_acc|+31.42|[+26.80, +36.04]|
|普通全表示保持|同池英语CE回放|IFEval／prompt_level_strict_acc|+20.33|[+16.08, +24.58]|
|同池英语CE回放|适配前模型|IFEval／prompt_level_strict_acc|-18.85|[-23.48, -14.23]|
|同池英语CE回放|新All-layer LoRA（batch2）|IFEval／prompt_level_strict_acc|+11.09|[+7.21, +15.16]|
|新All-layer LoRA（batch2）|适配前模型|IFEval／inst_level_strict_acc|-23.38|[-27.10, -19.77]|
|普通全表示保持|适配前模型|IFEval／inst_level_strict_acc|+1.44|[-0.63, +3.61]|
|普通全表示保持|新All-layer LoRA（batch2）|IFEval／inst_level_strict_acc|+24.82|[+21.37, +28.03]|
|普通全表示保持|同池英语CE回放|IFEval／inst_level_strict_acc|+16.31|[+13.08, +19.47]|
|同池英语CE回放|适配前模型|IFEval／inst_level_strict_acc|-14.87|[-18.06, -11.64]|
|同池英语CE回放|新All-layer LoRA（batch2）|IFEval／inst_level_strict_acc|+8.51|[+5.44, +11.52]|
|新All-layer LoRA（batch2）|适配前模型|Global-MMLU SW／ACC|-0.35|[-1.25, +0.49]|
|普通全表示保持|适配前模型|Global-MMLU SW／ACC|+1.96|[+1.13, +2.79]|
|普通全表示保持|新All-layer LoRA（batch2）|Global-MMLU SW／ACC|+2.31|[+1.63, +2.95]|
|普通全表示保持|同池英语CE回放|Global-MMLU SW／ACC|+1.59|[+1.00, +2.16]|
|同池英语CE回放|适配前模型|Global-MMLU SW／ACC|+0.37|[-0.48, +1.19]|
|同池英语CE回放|新All-layer LoRA（batch2）|Global-MMLU SW／ACC|+0.72|[+0.07, +1.37]|

## 复算与边界

- scores.csv包含40条绝对分数，paired_changes.csv包含48条配对比较；所有正负项均保留。
- numeric_items.json仅含匿名顺序下的逐题数值、权重和身份摘要，可在CPU复算，不含原题、回答或模型。
- 285条学科分解见gmmlu_subject_scores.csv；总体知识ACC按14042题加权，不是57学科宏平均。
- IFEval类型统计是描述性分解，不用于选方法；GSM8K严格／灵活提取差异包含输出格式因素，不自动等同纯推理差异。
- 仅SW、单阶段适配、单seed42，尚非顺序CL或跨语言方法泛化的证据；所有中间节点不参与Final选择。
- 本轮未重新测翻译或XNLI，历史结果保留，不能声称其问题已解决。
