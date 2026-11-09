# XQuAD补测

- 官方支持AR/RU，SW不支持；每语言1190题、240段文章。
- 测Base、历史LoRA及当前已存在/正在完成的对应Final400，共10个模型语言单元；四个未训练模型已按用户优先级明确暂缓。
- 标准LM Evaluation Harness生成式问答，0-shot、non-thinking、greedy、最多128新token、换行停止；原文直接按官方SQuAD v1.1 EM/F1评分，全部参考答案取最大值。
- 原始数据revision和提示/评分/模型身份在configs/xquad_20261109与freeze.json。XQuAD作为补充，固定5+5及TargetACC4不变。
- results.md、scores.csv、paired_changes.csv、score_audit.json持续更新；每题结果本机保留，置信区间以240段文章分簇重采样，不把同一段的题目当独立样本。
- 没有额外训练、没有测试选提示/选模型，全部完成后上传GitHub。正式得分持续写入 results.md；未完成单元格标为缺失，不代表零分。

完整10端点评测已结束，见 [完整报告](final_report.md)。`numeric_verification.json` 复算20项绝对分数和42项文章配对区间；主5+5仍在进行，不代表整体研究完成。
