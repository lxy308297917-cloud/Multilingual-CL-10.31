# WikiLingua：第二语言摘要收益与旧语言性能下降

七轨道主实验中，本轨道在冻结正式测试上同时出现可靠的TH摘要改善和同任务旧ES分数下降。这里的学习与遗忘按固定ROUGE-Lsum操作性定义，不据此宣称普遍语言知识增加或擦除。

## 数据、设置和评分

采用作者参与整理的GEM官方划分之唯一映射非空子集，不称全量官方test。ES21164、TH2723、FR12128、EN27004，共63019条/模型。训练候选四语共同语义ID3985，按训练前固定顺序选3200，ES/TH共享A ID；FR/EN仅监测。

训练排除所有已映射dev/test英语文章。dev/test不同section仍共享4685个英语文章URL，不声称文章级独立划分；测试按英语文章聚类做2000次配对bootstrap。对应输入、提示、参考与模型角色已核验，全部原始预测保留在本地。

每段200 optimizer steps、有效batch16、AdamW LR1e-5、BF16、assistant-only、non-thinking、长度2048、无packing，摘要microbatch1/accumulation8、双卡ZeRO3。SEQ共享Single ES Final，重置优化器和scheduler再训TH200步；逻辑累计400步，不能当作和Single TH相同总算力。

摘要采用训练前冻结的多语言ROUGE-Lsum v2句子分词修正，统一对同一份生成重新评分，R1/R2逐样本保持一致。没有改变测试生成或挑checkpoint。该修正版不是GEM原排行榜评分，见[来源补充](wikilingua_protocol_addendum.md)和[评分修正](summary_scoring_correction.md)。

# wikilingua：固定Final阶段矩阵

| Benchmark | 语言 | 样本数 | Base | Single ES | Single TH | ES→TH |
|---|---|---:|---:|---:|---:|---:|
| wikilingua | en | 27004 | 18.25 | 33.05 | 31.91 | 32.89 |
| wikilingua | es | 21164 | 17.34 | 30.14 | 27.29 | 28.85 |
| wikilingua | fr | 12128 | 17.73 | 27.29 | 27.18 | 27.30 |
| wikilingua | th | 2723 | 21.01 | 16.12 | 31.52 | 32.99 |

单位：ACC为百分数，摘要为ROUGE-Lsum F1分；不合并不同任务。ES遗忘为Single ES减SEQ，正值表示下降。

| Benchmark | 指标 | 变化及95%区间 |
|---|---|---:|
| en:wikilingua | ES阶段迁移 | +14.80 [+14.63, +14.98] |
| en:wikilingua | TH阶段变化 | -0.17 [-0.29, -0.05] |
| es:wikilingua | G_ES | +12.81 [+12.61, +13.00] |
| es:wikilingua | F_ES | +1.30 [+1.16, +1.43] |
| fr:wikilingua | ES阶段迁移 | +9.56 [+9.29, +9.82] |
| fr:wikilingua | TH阶段变化 | +0.01 [-0.18, +0.20] |
| th:wikilingua | Single TH收益 | +10.51 [+9.90, +11.15] |
| th:wikilingua | G_TH | +16.86 [+16.23, +17.51] |
| th:wikilingua | T_TH | -4.89 [-5.29, -4.49] |
| th:wikilingua | SEQ Final−Single TH Final | +1.46 [+1.03, +1.91] |
| th:wikilingua | 阶段收益差（非因果） | +6.35 [+5.77, +6.96] |

## 核心观察

- Single ES把西语摘要从17.34提高到30.14；Single TH把泰语从21.01提高到31.52，目标匹配任务具备学习信号。
- ES后未训练泰语降至16.12，相对Base为-4.89[-5.29,-4.49]。TH阶段升至32.99，阶段增益16.86[16.23,17.51]。较大的阶段增益同时受到较低起点影响，应与Single TH正对照一起理解。
- 旧西语30.14→28.85，下降1.30[1.16,1.43] ROUGE-Lsum分，区间不跨零。但Final仍高于Base17.34，是部分同任务性能回退，不是旧语言能力全部丢失。
- SEQ相对Single TH高1.46[1.03,1.91]；这是当前不同累计预算、Same-ID语义内容重复条件下的终点差异，不是同预算算法优势或纯语言切换因果。
- 英语同任务摘要TH阶段下降0.17分[0.05,0.29]，法语变化区间跨零；其他任务指标分开报告。

# wikilingua：统一旧能力监测

单位：百分数。固定Final，不依据这些正式分数早停或选模型。GSM8K严格格式与灵活数值提取分列；IFEval按官方prompt/instruction strict为主，loose另留JSON。

| Benchmark / 指标 | n | Base | Single ES | Single TH | ES→TH |
|---|---:|---:|---:|---:|---:|
| gsm8k:exact_match:flexible-extract | 1319 | 73.16 | 56.71 | 58.00 | 35.94 |
| gsm8k:exact_match:strict-match | 1319 | 44.81 | 6.14 | 0.00 | 0.00 |
| ifeval:inst_level_strict_acc:none | 834 | 74.94 | 49.52 | 46.76 | 37.65 |
| ifeval:prompt_level_strict_acc:none | 541 | 66.36 | 36.60 | 32.35 | 25.14 |
| reading:acc:none | 900 | 60.89 | 36.44 | 75.11 | 28.89 |

| Benchmark / 指标 | ES→TH阶段变化及95%区间 | 相对Base变化及95%区间 |
|---|---:|---:|
| gsm8k:exact_match:flexible-extract | -20.77 [-23.65, -17.89] | -37.23 [-40.03, -34.12] |
| gsm8k:exact_match:strict-match | -6.14 [-7.43, -4.93] | -44.81 [-47.46, -42.15] |
| ifeval:inst_level_strict_acc:none | -11.87 [-15.06, -8.39] | -37.29 [-41.61, -32.97] |
| ifeval:prompt_level_strict_acc:none | -11.46 [-15.16, -7.58] | -41.22 [-46.40, -36.22] |
| reading:acc:none | -7.56 [-11.00, -4.22] | -32.00 [-36.33, -27.78] |

## 输出长度、语言和阅读读出

[完整输出诊断](../../results/unified_language_cl/wikilingua/output-audit-v1/table.md)复用相同生成记录：

- 泰语ES后平均420.9 token，77.01%达到生成上限；TH Final平均138.0 token，16.27%达到上限；参考平均89.5 token。TH阶段ROUGE恢复伴随强烈的长度控制变化，不能把全部增益解释为新增语义理解。
- 西语ES后平均58.9 token，SEQ后43.7，参考65.1；输出缩短及语言读出变化可能参与ROUGE下降。固定协议仍测得真实任务表现回退，但尚未分离长度、语言混杂和内部知识变化。
- 泰语源截断22.99%，各模型一致。自动语言识别在Final上的覆盖95.19%、正确/全部93.28%、正确/有置信97.99%；这是langdetect估计，不是人工语言正确率，更不是内容忠实性评估。
- 通用GSM8K flexible在TH阶段下降20.77个百分点，IFEval prompt strict下降11.46个百分点，EN阅读下降7.56个百分点；对应原有配对区间均不跨零。
- EN阅读是四选项log-likelihood ACC，不是自由生成字母解析。逐题复核与原分数一致；Final预测A/D合计95.78%，而Single TH阅读75.11%且选项更分散。[完整22模型读出复核](../../results/unified_language_cl/en-reading-choice-audit-v1/report.md)说明选项集中值得区分，但不据此确认因果机制或校准新的测试成绩。

## 研究范围和下一步

同任务TH改善、旧ES下降以及部分通用任务退化在本轨道同时出现，是可用于后续研究的基线现象。当前只得到固定模型上的行为证据：ROUGE不能单独证明语义忠实性，选项分布不能单独证明知识擦除，单seed区间不涵盖训练随机性，未做多重检验校正。

3985共同训练ID不足两组各3200的原预算Disjoint-ID对照，本轮不事后缩预算补出有利结果。P3按原资格在MASSIVE、XNLI、MTOP继续；本轨道的重复内容、累计预算和任务读出混杂保留为限制。

正式测试未用于改变学习率、样本、预算、解析或checkpoint。未运行新方法、额外种子、FR第三阶段或选项置换干预。整个项目仍需完成预定中点和三个合格轨道P3对照，再给出最终研究结论。
