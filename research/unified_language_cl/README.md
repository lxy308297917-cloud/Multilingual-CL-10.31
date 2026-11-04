# ES→TH 多任务持续学习基线

建立可信的Single FFT与FFT-SEQ基线，记录学习、迁移、旧语言保持和通用任务变化。本轮不开发My Method，不运行SST、Replay、FR第三阶段或额外种子。

## 当前进度

按固定Final完成凭证及四语言、通用能力矩阵核对；启动或进程存活不代表测评完成。

| 轨道 | 主训练段 | 原生正式矩阵 | 通用能力矩阵 | 完整原生结果 | 通用结果 |
|---|---:|---|---|---|---|
| massive | 3/3 | 完成 | 完成 | [矩阵](../../results/unified_language_cl/massive/comparison-test-v1/table.md) | [矩阵](../../results/unified_language_cl/massive/retention-comparison-v1/table.md) |
| mathoctopus | 3/3 | 完成 | 完成 | [矩阵](../../results/unified_language_cl/mathoctopus/comparison-test-v1/table.md) | [矩阵](../../results/unified_language_cl/mathoctopus/retention-comparison-v1/table.md) |
| mtop | 3/3 | 完成 | 完成 | [矩阵](../../results/unified_language_cl/mtop/comparison-test-v1/table.md) | [矩阵](../../results/unified_language_cl/mtop/retention-comparison-v1/table.md) |
| xnli | 3/3 | 完成 | 完成 | [矩阵](../../results/unified_language_cl/xnli/comparison-test-v1/table.md) | [矩阵](../../results/unified_language_cl/xnli/retention-comparison-v1/table.md) |
| xlsum | 3/3 | 完成 | 完成 | [矩阵](../../results/unified_language_cl/xlsum/comparison-test-v1/table.md) | [矩阵](../../results/unified_language_cl/xlsum/retention-comparison-v1/table.md) |
| wikilingua | 3/3 | 进行中 | 进行中 | 进行中 | 进行中 |
| sib200 | 3/3 | 完成 | 完成 | [矩阵](../../results/unified_language_cl/sib200/comparison-test-v1/table.md) | [矩阵](../../results/unified_language_cl/sib200/retention-comparison-v1/table.md) |

主训练21/21段；完整轨道6/7。整个目标继续进行，正式矩阵齐全后还须完成预定开发节点、三个合格轨道P3及最终交付核验。

## 结果与复核入口

- [六轨道阶段总表](six_track_summary.md)：保留该次发布范围，不冒充七轨道最终结论。
- [MTOP解释](mtop_results.md)、[XNLI解释](xnli_results.md)、[XL-Sum解释](xlsum_results.md)、[SIB-200解释](sib200_results.md)。
- [数学负结果与输出错误复核](math_output_diagnostics.md)。
- [固定开发64题分类读出复核](../../results/unified_language_cl/classification_readout_complete_v2/table.md)。
- [摘要评分修正](summary_scoring_correction.md)：所有摘要Base/Final使用同一v2句子分词，旧分数保留，不把评分修正当训练收益。
- [数据审计](dataset_audit.md)、[语言对齐与污染限制](language_alignment.md)、[WikiLingua来源补充](wikilingua_protocol_addendum.md)。
- [冻结测评协议](evaluation_protocol.md)、[复现步骤](reproduction.md)、[完整任务书](task_specification.md)。

## 固定实验设置

官方后训练Qwen3-1.7B，revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。每轨道分别训练Base→ES、Base→TH、共享ES Final→TH。每段3200例/200 optimizer steps，SIB-200为640例/40步；有效batch16，seed42，AdamW LR1e-5，BF16，assistant-only、non-thinking，长度2048、无packing。SEQ第二阶段重置优化器及scheduler，固定Final，不因英语下降早停。

Single TH与SEQ的TH实际输入/标签顺序一致，但SEQ累计两段预算，不能当相同总算力。FR和EN仅监测；不同任务的ACC、Tree EM及ROUGE不混合平均。

## 当前结论的边界

- 已完成轨道中，多项任务能学习目标任务，同时部分通用能力退化；MathOctopus数学结果低于Base，负结果保留。
- 尚无可靠的同任务旧西语下降证据；英语阅读有升有降，不概括为统一EN遗忘分数。
- 标签读出、摘要长度控制和窄任务训练都会影响行为成绩。现有分数不单独证明语言语义知识增加或内部通用知识擦除。
- 所有配对区间条件于seed42固定模型，未做多重检验校正，不是训练种子方差；区间跨零不等于等价。

## 下一步：按原计划完成

WikiLingua官方唯一映射测试子集每模型63,019条生成仍在进行；固定中点开发评测衔接在主正式矩阵后。随后对MASSIVE、XNLI、MTOP各运行预留Disjoint-ID的Single TH和SEQ TH，共六段，沿用原预算及正式协议。

[P3资格与范围](p3_eligibility.md)在正式对照之前冻结。MathOctopus、WikiLingua与SIB-200不足原预算所需共同ID，XL-Sum非平行，不事后缩预算造对照。Same-ID与Disjoint-ID仍有内容难度差异，不能直接视为纯语言切换因果。

公开交付只包括代码、来源/配置锁与聚合结果；原始问题、摘要、完整预测、训练缓存和模型权重不在本仓库再分发。
