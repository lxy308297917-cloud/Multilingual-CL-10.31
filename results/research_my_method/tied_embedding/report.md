# tied embedding有限恢复：探索性结果

1710条预测的gold、argmax、同模型baseline精确logits及四个实际范数控制独立复核通过。

| 条件 | EN知识dev ACC | SW知识dev ACC | HA知识dev ACC |
|---|---:|---:|---:|
| unchanged | 62.28% | 31.58% | 28.95% |
| restore_tied_embedding_alpha0.5 | 58.77% | 32.46% | 29.82% |
| restore_tied_embedding_alpha1.0 | 58.77% | 31.58% | 31.58% |
| random_sign_matched_alpha0.5 | 62.28% | 31.58% | 28.95% |
| random_sign_matched_alpha1.0 | 62.28% | 30.70% | 29.82% |

## 预登记主要对比：完整恢复减同范数随机符号

| 语言 | ACC差pp | 95%配对区间 |
|---|---:|---:|
| en | -3.51 | [-9.65, +2.63] |
| sw | +0.88 | [-8.77, +10.53] |

不能据96.21%能量直接称embedding造成全部遗忘。这里的EN知识在适配后本就较Base提高，不是发生IFEval/GSM遗忘的同一任务。恢复未显示明确EN/SW改善，随机符号扰动也不代表完整语义空对照。输入/输出角色仍绑在一起；不是新方法或正式benchmark。
