# 统一语言顺序CL：首批固定预算结果（进行中）

本轮目标：七个独立任务轨道的ES→TH Single FFT / FFT-SEQ基线，FR只测评。相同官方后训练Qwen3-1.7B锁定revision，seed42，3200条/200步/段，effective batch16，AdamW LR1e-5，cosine warmup0.05，BF16，长度2048，assistant-only non-thinking，不packing。SIB-200为640条/40步。Final固定，不按EN保持或正式test选checkpoint。

**这次提交是首批结果快照，不是整个目标完成交付。** MASSIVE、MathOctopus各三段训练和四模型正式矩阵已完成；另五轨数据均已准备，MTOP正式训练与通用能力评测正在继续。新训练/评分代码和完整配置将在最终交付统一整理，本次只发布聚合结果、协议和图。

## 已观察到什么

- MASSIVE：ES→TH使TH从76.60%升至81.34%；ES从84.03%变为83.83%，下降0.20个百分点的配对区间跨零。不能称已有稳定遗忘。标签与格式学习对收益有贡献；固定同题64题候选诊断单列。
- MathOctopus：正式MGSM-Rev2和MSVAMP相对Base四语均下降；不能称目标任务学习成功。训练来源开发曲线中的恢复不能替代正式测试收益。####格式与数值正确率分别报告。
- **MASSIVE完整通用能力监测已完成：** ES→TH最终GSM8K flexible为67.17%（Base73.16%，下降5.99个百分点），IFEval prompt strict为44.18%（Base66.36%，下降22.18个百分点）；英语阅读由60.89%升至79.44%。因此已观察到目标任务收益与部分通用能力退化，并非英语所有能力都下降。ES旧语言遗忘仍不明确。
- MathOctopus：GSM8K strict由44.81%升至57.24%，但flexible由73.16%降至57.39%；严格格式分数改善不能等同于数学能力提高。IFEval和阅读也下降。
- 首批6模型与Base的统一EN监测均已完成。其余五轨继续；所有负结果保留，不据此宣布整个目标已完成。

## 结果入口

- [MASSIVE正式阶段矩阵与区间](../../results/unified_language_cl/massive/comparison-test-v1/table.md)
- [MathOctopus正式阶段矩阵与区间](../../results/unified_language_cl/mathoctopus/comparison-test-v1/table.md)
- [同题开发64题生成/候选读出诊断](../../results/unified_language_cl/first_batch_readout_v1/matched_dev64.md)
- [MASSIVE统一通用能力与区间](../../results/unified_language_cl/massive/retention-comparison-v1/table.md)
- [MathOctopus统一通用能力与区间](../../results/unified_language_cl/mathoctopus/retention-comparison-v1/table.md)
- [实际训练曝光和权重验证](../../results/unified_language_cl/first_batch_training_receipt.json)
- [开发学习曲线](../../results/unified_language_cl/first_batch_dev_curves_v1/learning_curves.png)
- [冻结协议及原始Wiki来源阻塞说明](evaluation_protocol.md)
- [WikiLingua官方来源恢复补充](wikilingua_protocol_addendum.md)

## 解释边界

只有seed42，配对区间条件于当前已训练模型，不覆盖训练随机性，未做多重检验校正。分类、结构、数学、摘要不混合成一个总分。先ES再TH与直接TH的总训练量不同，阶段收益差不解释为纯语言因果。MathOctopus同题ID经过英文源题和数值参考一致性重建，不等于人工语义认证；有效训练候选6059不足6400，不伪造Disjoint比较。XL-Sum为非平行原生新闻。

WikiLingua采用作者支持的GEM官方发布，通过原文和摘要精确匹配原始英语URL/section；共同候选3985，训练共用3200。正式范围为唯一映射子集，不是全官方test。官方dev/test不同section共享部分文章；训练联合排除已映射评测文章，区间按原始文章聚类。图像语义对齐继承发布者映射，未独立重建图片对齐；不作更强的严格翻译或文章独立性声明。评分复用锁定XL-Sum多语言ROUGE，不冒充GEM排行榜评分。原始文本遵循相应来源许可，本仓库本次不再分发原文或全文预测。

公开快照不含GPU/PID、宿主目录、完整环境清单、原始逐题文本、训练缓存、权重或优化器。评分源码和数据/结果资产SHA随最终代码交付提供。
