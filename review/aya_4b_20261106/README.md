# Qwen3-4B 本轮全文纠错包

这是用户要求公开的训练和测评全文快照。保留全部文本，不用省略号替代重复回答；不改实验原件和冻结分数。未包含模型权重、token-ID数组、服务器进程或环境凭据。

## 从哪里开始

- **训练集**：[原6400条](training/main/train6400.jsonl.gz)及[来源逐行清单](training/main/train6400.provenance.jsonl.gz)；[重平衡6400条](training/translation_rebalanced/train6400.jsonl.gz)及[来源逐行清单](training/translation_rebalanced/train6400.provenance.jsonl.gz)。两个重平衡LoRA候选共用后一套数据。
- **直接浏览翻译训练句对**：[原640组EN→SW](training/main/mt_pairs.jsonl)、[重平衡1600组EN→SW](training/translation_rebalanced/mt_pairs.jsonl)。原完整训练集另含反向条目；两个方向共享原始句组。
- **完整MT输出**：`mt/base/`、`mt/fft/`、`mt/lora/`、`mt/ssu/`，每模型两个方向各1012句，每片最多128条。每条包含源句、参考译文、完整raw_output和prediction、生成长度和结束原因。
- **MT与其他任务的完整输入／实际提示**：`inputs/<benchmark>.jsonl.gz`。用review_id连接模型输出。提示中的few-shot和所有正文完整保留。
- **其他benchmark输出**：`outputs/<model>/<benchmark>.jsonl.gz`，包含SW/EN摘要、Belebele、SW Global-MMLU、EN MMLU、GSM8K、IFEval、XNLI SW/EN、XCOPA SW。
- **翻译开发对照**：`development/`保存原LoRA解码对照、重平衡LR1e-4、重平衡LR2e-5，各512条；与正式测评分目录，不混为正式成绩。
- `manifest.json`列出文件、行数、SHA256和覆盖范围；`source_hashes.json`记录导出来源文件哈希。

## 如何看选择题

Belebele、MMLU/Global-MMLU、XNLI/XCOPA采用候选答案似然评分，不存在自由生成的完整解释。inputs保留实际doc、scored_prompt、choice_continuations和标准答案；outputs保留每个选项真实分数、预测选项、正确性。不要把predicted_choice误称为自由生成回答。

GSM8K同一生成的strict/flexible提取结果保存在filter_results；IFEval保留当次官方逐指令及逐prompt判断。所有输入均跨四模型逐项身份一致性核验。

## 纠错方法

请把修改建议记录在corrections_template.csv的副本，注明任务、模型、review_id或训练行号、原文、建议及理由。训练的row_index为从0开始的行号；source_id用于定位原公开来源。请保留原始快照，纠错与是否重算另行记录。

本包含已暴露benchmark题目和答案，用于错误分析。既有冻结分数保持不变，不把基于这些test答案的调整冒充新的盲测改善。数据保留上游来源和revision；各来源数据许可沿用其原许可，不因仓库代码许可证而统一改许可。

## 这次导出的范围

四个固定模型：原Qwen3-4B、FFT Final400、LoRA Final400、SSU-Wanda Final400。使用canonical generation-v2与完成的HF/supplemental测评；不混入旧generation-v1或失败尝试。训练原文件及provenance经gzip解压逐字节验证；生成文本原样保留。源文件、模型和训练配置没有修改，也没有新增GPU任务。
