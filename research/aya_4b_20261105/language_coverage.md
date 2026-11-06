# 补充评测的语言覆盖与复用

本表只确认XNLI/XCOPA公开任务覆盖；不是对所有训练数据或完整评测栈的可用性承诺。首轮仍为SW，不启动其他语言训练。

|语言|XNLI|XCOPA|下一步用途|
|---|---|---|---|
|斯瓦希里语 SW|有|有|当前首轮|
|泰语 TH|有|有|后续候选；还需核实翻译/阅读/知识和训练资源交集|
|中文 ZH|有|有|后续候选；需明确简繁、脚本与语言代码|
|越南语 VI|有|有|后续候选；需核实原生任务及数据配方来源|
|土耳其语 TR|有|有|后续候选；不在本轮自动扩展|
|英语 EN|有|无XCOPA-en配置|因果常识参照需另锁定原COPA；不得把不同来源分数视为完全平行跨语言比较|
|法语 FR / 西语 ES|有|无|不适合作为要求XNLI+XCOPA都覆盖时的首批语言|

XNLI的15种语言与XCOPA的11种语言交集为SW、TH、ZH、VI、TR。所有语言使用同一种能力定义和主要ACC指标；不能每种语言换一个容易提升的任务。数据来源、split、样本数、few-shot、标签映射、chat模板、likelihood算法、解码及评分版本均在测评前锁定。

本机已找到官方facebook/xnli版本 `b8dd5d7af51114dbda02c0e3f6133f332186418e` 的缓存；官方all_languages test为5010条，validation为2490条。历史其他语言研究可能已读取该数据，因此后续开发需记录曝光，不称为全新盲测。XCOPA SW公开validation100、test500条；本机数据缓存与具体revision尚待核实，当前NOT_RUN。

数据来源：

- https://github.com/facebookresearch/XNLI
- https://huggingface.co/datasets/facebook/xnli
- https://huggingface.co/datasets/cambridgeltl/xcopa/blob/main/README.md

新补充结果将与旧正式表分开，并保留无提升和退化项。只有补充任务提高时，结论应限定为相应推理/常识能力；不得据此声称翻译或Global-MMLU已改善。
