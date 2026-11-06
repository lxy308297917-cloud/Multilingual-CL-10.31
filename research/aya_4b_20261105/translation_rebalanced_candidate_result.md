# 翻译数据重平衡候选：固定 Final400 开发结果

本次候选已完成，仍不构成正式翻译修复证据。原四模型结果不改。

## 训练核验

6,400 条、400 optimizer steps、effective batch16，seed42；LoRA rank16/alpha32/dropout0.05，LR1e-4，cosine/warmup0.05。实际输入1,328,744 tokens，监督209,925 tokens。全部6,400条输入/标签曝光指纹与冻结候选一致，每步16条，80条loss/梯度与学习率日志均核验。训练器时间2571.88秒，训练进程时间2579.12秒。最终adapter哈希见审计文件；未保存optimizer恢复状态。

## 冻结开发对照

同64个已暴露FLORES dev句组、双向、同一批Base重跑；不是正式test。官方non-thinking采样采用预定T0.7/top_p0.8/top_k20/min_p0/seed42/max512。

|方向|同协议Base chrF++|新LoRA chrF++|差异|配对95%区间|
|---|---:|---:|---:|---|
|EN→SW|16.82|17.95|+1.13|[-2.79,+4.90]|
|SW→EN|35.58|36.05|+0.47|[-1.93,+2.74]|

原greedy列也完整保留：EN→SW新LoRA10.17、Base14.27，仍低4.10分；不能只展示采样改善。相同Base的旧开发运行与本次重跑有小幅差异，只用本次配对结果。

## 判定与边界

预先规定的EN→SW差异区间下界>0条件未达到。点估计小幅为正，但证据不足以确认修复；也不足以断言效果为零。bootstrap区间为当前固定模型与一次推理输出的句组采样不确定性，不代表跨训练或推理种子稳健性。

在本候选分析时，第二个同数据LR2e-5候选尚未运行；现已完成且未达到开发条件，见[第二个候选结果](translation_lr2e5_candidate_result.md)和[最终研究结论](final_decision.md)。新候选正式FLORES、阅读、知识、摘要、数学及IFEval：NOT_RUN，不能推断保持或改善。知识数据源未专门更换，不能据此声称Global-MMLU问题已解决。

数据重平衡同时减少长摘要并增加平行翻译覆盖，且监督token总量改变；不能分离其独立因果作用。后续语言拓展仍应复用同一翻译、阅读、知识及能力保持测评框架，当前未训练其他语言。

[开发分数、配对区间与成本](../../results/aya_4b_20261105/translation-rebalanced-development-analysis-v1/results.json)

[完整开发表](../../results/aya_4b_20261105/translation-rebalanced-development-analysis-v1/report.md)

[训练曝光和权重审计](../../results/aya_4b_20261105/translation-rebalanced-training-audit-v1.json)
