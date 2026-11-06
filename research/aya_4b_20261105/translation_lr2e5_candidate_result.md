# 第二个翻译修正候选：LR2e-5

此候选为预定有限方案的第二组，保持重平衡数据及其余训练设置，固定Final400。原四模型正式表及第一候选不改。开发材料已暴露，不能冒充正式确认。

|方向|相对同协议Base chrF++|配对95%区间|
|---|---:|---|
|en->sw|-2.29|[-5.17,+0.22]|
|sw->en|-0.31|[-2.58,+1.77]|

冻结开发继续投入条件：未达到；停止扩大该有限搜索，保留负面和不确定结果。

新候选正式test和各保持任务：NOT_RUN。未训练第三个候选、新种子、其他语言或MyMethod。

[完整开发对照及统计](../../results/aya_4b_20261105/translation-rebalanced-lr2e5-development-analysis-v1/results.json)

[训练曝光和权重审计](../../results/aya_4b_20261105/translation-rebalanced-lr2e5-training-audit-v1.json)
