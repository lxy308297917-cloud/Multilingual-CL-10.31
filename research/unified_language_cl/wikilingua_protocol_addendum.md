# WikiLingua 官方来源恢复与冻结补充协议

原始 `P0_GATE.json`、Drive 404记录及最初协议保留；本补充在任何WikiLingua模型评分之前冻结。

## 来源与对齐

采用作者参与整合的 [GEM WikiLingua 数据卡](https://gem-benchmark.com/data_cards/wiki_lingua) 对应 `GEM/wiki_lingua@af5d0f00b59a6933165c97b384f50d8b563c314d`。2335323394字节的规范档案已与上游LFS SHA `95663019cebc422c9b017f6b670a67a272eef259ce69d63677bf4606b692ed83` 一致。未执行远程数据加载代码；仅从白名单tar普通文件读取，反序列化器拒绝所有全局类/函数。

通过规范化文档和摘要的精确二元组匹配原始发布者的英语URL与英语section字段；不按GEM通用编号或行顺序对齐。原始英文使用自身URL与section_name。歧义、无匹配和空内容不进入映射子集，筛选不依赖任何模型输出。

训练候选排除所有语言已映射dev/test中的英语文章URL后，四语共同ID为3985；SHA排序选3200，同一语义顺序供ES和TH阶段使用。FR/EN不训练。开发集四语共同ID中选256，各模型固定复用。训练预算200步、effective batch16不变。

## 正式评测范围

| 语言 | 官方原生test条数 | 唯一映射非空test条数 |
|---|---:|---:|
| ES | 21726 | 21164 |
| TH | 2876 | 2723 |
| FR | 12405 | 12128 |
| EN | 27489 | 27004 |

结果命名为“GEM WikiLingua官方划分的唯一映射子集”，不能称全量官方test。官方section-ID跨划分无重叠，但dev/test不同section共享4685个英语文章URL。因此不能声称文章级独立dev/test；训练已联合排除对应已映射文章。未知/歧义映射的覆盖限制如实保留。

## 生成和评分

沿用已冻结摘要协议：完整任务及assistant头内保留最长正文token前缀，使源长≤1536；训练监督≤512、总长≤2048。正式参考不截断。non-thinking BF16贪心生成，max_new_tokens512。

采用锁定XL-Sum作者多语言ROUGE（ES/FR/EN Snowball、TH PyThaiNLP newmm），主指标ROUGE-Lsum F1，附ROUGE-1/2、长度、截断覆盖、辅助语言诊断。该评分不是GEM排行榜原评分。四语言自同一与空输出oracle已验证。

许可遵守CC-BY-NC-SA-3.0，不公开再分发原文、全文逐题预测或训练缓存。受限CPU审计、训练及评分代码、哈希与聚合结果可交付。候选3985不足6400，本轨不运行原预算下的Same/Disjoint对照。
