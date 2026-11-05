# 七轨道固定开发学习曲线

全部预定中点已完成：最先两条轨道6个中点，加上其余五条15个中点。Base、每条独立Single ES/Single TH及共享ES Final→TH的中点和固定Final，共196个语言分数点。

- [PNG全图](../../results/unified_language_cl/all_dev_curves_v1/learning_curves.png)
- [PDF全图](../../results/unified_language_cl/all_dev_curves_v1/learning_curves.pdf)
- [可复核CSV](../../results/unified_language_cl/all_dev_curves_v1/learning_curves.csv)
- [逐来源文件SHA和输入一致性凭证](../../results/unified_language_cl/all_dev_curves_v1/verification.json)

每条轨道、每种语言的所有点复用同一冻结开发ID、prompt-token哈希、参考和数学数值答案。分类ACC、MTOP Tree EM与摘要ROUGE-Lsum各自保留单位，摘要另设轴范围，不跨任务平均。

横轴是累计optimizer steps。普通轨道ES阶段200、TH阶段200；SIB-200为40+40。Single TH只使用一段预算，不能将SEQ终点和Single TH当作同总算力方法。中点是原200步（SIB40步）调度中的保存状态，不是重新训练的独立100步预算实验。

这些点仅描述固定模型轨迹，没有训练种子误差条，没有据此挑选Final或提前停止。MathOctopus的dev来自训练源题独立留出，不是MGSM/MSVAMP正式test；正式测试另列在[阶段矩阵](stage_language_matrix.md)。

整体可见分类和解析较快获得任务表现，数学内部dev在第一段下降、后续部分恢复，摘要ES后泰语下降而TH阶段恢复。这些形状仍混合任务格式、长度控制与语言因素，不能只用曲线推导参数冲突机制。正式Final的配对区间、输出复核及P3对照分别报告。

主实验及学习曲线已完成，MASSIVE/XNLI优先P3对照正在训练与测评；MTOP对照与最终研究结论尚未完成。
