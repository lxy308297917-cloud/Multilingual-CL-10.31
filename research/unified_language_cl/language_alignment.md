# 语言与语义身份：可支持的比较范围

| 轨道 | 同源训练候选 | 对齐依据 | 原预算Disjoint-ID可行性 |
|---|---:|---|---|
| massive | 11514 | Official id maps back to original SLURP semantic item; labels and partitions checked across languages. | 可行，冻结额外TH3200条；尚须完成P3实跑 |
| mathoctopus | 6059 | RECONSTRUCTED English-led blocks, exact official GSM8K-train question identity and numeric agreement in all four languages. Not released semantic IDs; numeric agreement does not prove translation semantics. Same-ID interpretation qualified. | 不可行：不降低原预算凑对照 |
| mtop | 7896 | Official v2 IDs explicitly identify translations (release README). Native compositional decoupled column7, whitespace canonicalized by parser; approximate parallelism, not perfect translation semantics. | 可行，冻结额外TH3200条；尚须完成P3实跑 |
| xnli | 392484 | Shared multilingual object in official all_languages parquet. Unique annotation ID = split + joint-object row + hash(English premise,hypothesis); semantic group=English text hash. No independent language-row zipping. Official dev/test annotations retained, even repeated texts. | 可行，冻结额外TH3200条；尚须完成P3实跑 |
| xlsum | 0 | NONPARALLEL native BBC articles. No cross-language ID alignment or pure language causality. | 非平行，不定义Same-ID伪对照 |
| wikilingua | 3985 | Exact normalized source+summary matched to original publisher English URL+section; shared IDs for ES/TH/FR/EN training and dev. Never aligned by row position. | 不可行：不降低原预算凑对照 |
| sib200 | 701 | Official index_id, categories checked across languages. | 不可行：不降低原预算凑对照 |

## 来源和身份限制

- 官方ID或联合对象提供来源身份，不能认证全部翻译语义质量；MTOP仅近似平行。MathOctopus重建英文题块和官方GSM8K train精确题目身份并核对数值，不按答案相同或独立行号对齐。
- WikiLingua训练排除所有已映射dev/test英语文章，训练候选3985；dev/test不同section仍共享4685个英语文章URL。因此测试统计按英语文章聚类，不能声称dev/test文章级独立。详见[来源补充](wikilingua_protocol_addendum.md)。
- XL-Sum是各语言独立新闻，所有模型在同一种语言内部对相同测试文章配对；不对不同语言文章做虚假的逐题配对。
- MASSIVE、XNLI、MTOP的额外B样本在评分前预留、与A源ID不重叠；不保证语义新颖或题目难度相同。MASSIVE/MTOP仍分别有38/25条B的规范化TH文本在A中出现，明确保留不重新挑样本。见[对照资格](p3_eligibility.md)。
- 精确去重与交叉划分检查不等于排除语义污染或预训练曝光。XL-Sum EN、GSM8K、IFEval、EN阅读存在既往项目使用，不称全新盲测；本轮冻结后不依成绩调参。
- 机器可读审计保留训练前test曝光声明，因此出现NOT_RUN是历史冻结状态，不是当前进度；当前完成情况看阶段矩阵与队列完成凭证。
- 许可按来源文件声明记录；不重新分发原始题目、目标树、摘要全文、逐题预测或训练缓存。来源revision/资产SHA见[sources lock](../../configs/unified_language_cl/sources.lock.json)，选定数据的protocol SHA见[本轮审计](../../results/unified_language_cl/data-audit-v2/audit.json)。
- FR/EN只做监测，不训练；本轮无FR第三阶段，也不据分数改语言顺序。
