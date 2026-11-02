# CrossSum 数据可行性

状态：**NOT_RUN（P1，仅完成数据与重合审计）**。

官方来源：[CrossSum](https://github.com/csebuetnlp/CrossSum)，[固定数据源](https://huggingface.co/datasets/csebuetnlp/CrossSum/tree/b3fb19e9c135397c65f11fc9cf4ce58bf86aa29b)。归档版本1.0.0，HF revision `b3fb19e9c135397c65f11fc9cf4ce58bf86aa29b`。

| 方向 | 官方test条数 | 唯一配对ID | 与XL-Sum test共享源URL/原文 | 共享目标URL/摘要 |
|---|---:|---:|---:|---:|
| EN→SW | 270 | 270 | 6 | 37 |
| SW→EN | 270 | 270 | 37 | 6 |

配对ID为`SHA256(source_url + "|" + target_url)`，无空原文。共享URL和精确文本匹配数量相同。比较范围仅为本轮XL-Sum官方test，不包含其train/validation，不能把表中数量说成全库总重叠。

数据有明确方向和test划分，可以重建；但检索配对并非人工逐对语义等价认证。此次未获得可靠的独立双语语义质量标注，因而不把它作为正式跨语言摘要证据。这个决定表示质量尚未验证，不表示已经证明数据质量差。先保证P0全量测评完成，保留配置和数据SHA供后续独立质量核验。

即使以后通过质量核验并运行，它也不能作为与XL-Sum完全独立的测试重复计数。
