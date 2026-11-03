# 第二语言选择与数据审计

选择状态：**Hausa已选定并完成训练输入冻结**；普通FFT第二阶段400步及其全部固定正式比较已完成；SST-NoRefresh第二阶段正在训练。语言选择发生在任何候选语言模型正式分数读取之前。

| 项目 | Hausa | Yoruba | Urdu |
|---|---:|---:|---:|
| 人工Aya Dataset条数 | 3,512 | 11,758 | 654 |
| 输入/答案精确唯一对数 | 3,408 | 6,673 | 646 |
| 人工Aya字符长度中位数 | 114 | 1,411 | 750.5 |
| 同锁Aya Collection训练分片 | 850.1MB，1片 | 约1.026GB，2片 | 约1.449GB，2片 |
| 官方Global-MMLU配置 | ha，存在 | yo，存在 | 不存在 |
| 官方Belebele | hau_Latn | yor_Latn | urd_Arab |
| 当前锁BenchMAX标准IFEval | 无 | 无 | 无 |

人工Aya与既有SW6400使用的Aya Collection是不同数据源。上表不能将人工Aya数量当作Collection数量。Hausa与SW使用拉丁文字，语言谱系不同；该差异不能单独保证更大遗忘。

## Hausa实际Collection审计

CohereLabs/aya_collection_language_split，revision `a3af2fde4b4cb5b2775830b11244a1a20b5f004f`，hausa/train，3,608,883条。完整850,122,119字节文件SHA256与Hub LFS一致：`49b6e343fa04638cc0a4e4d0e9ad596a629c1db8bbe3177d0e20a183ac174f1e`。下载截断和单个范围超时记录均保留，定点补齐后才使用。

以原SW6400的11个来源家族为固定白名单，每家族最多1400条蓄水抽样。共14,224条候选，经与SW相同的长度、损坏字符、指令精确去重及GlotLID检查，13,199条合格。未做严格全库去污染扫描。

合格数量：PIQA1389、MasakhaNEWS1380、AfriQA220、人工Aya1271、PAWS1387、Dolly1244、SODA1380、AdversarialQA1224、HotpotQA1366、CNN/DailyMail1234、WikiLingua1104。足以匹配原SW6400逐来源配额；AfriQA余量较小，开发预留须显式核算，不能补样或重复凑数。

选Hausa的理由：已有同版本实际源文件、相同来源家族配额可满足、阅读与知识任务覆盖完整、官方开发题可按共同语义ID核查。Yoruba仅完成人工Aya及Collection元数据审计，尚未进行同等全分片检查；不宣称其质量更差。Urdu主要受当前知识评测与人工数据规模限制。

## 配对和评测边界

Global-MMLU官方dev的EN/SW/HA/YO各285题，全部共同sample_id、subject、正确选项一致，已核对非空问题和四选项。每个subject5题，按固定SHA顺序3fit、2confirmation，得到171/114个语义ID。确认集已用于一次预登记有限干预及固定阶段开发比较；EN/SW整套dev还曾用于旧checkpoint选择/上下文示例，不能称为历史未见。

共同ID及正确字母一致只证明结构配对，不证明译文无误。没有自行机器翻译答案，也没有把dataset的文化标注字段误当作译文质量认证。当前尚缺独立双语语义复核；因此允许做探索性、任务形式一致的诊断，暂不能将这套材料作为“新方法已有效”的确认依据。

Hausa阅读、知识可以沿官方任务报告ACC。当前没有该语言的标准BenchMAX IFEval，不自制翻译规则后冒称官方成绩。EN/SW的既定IFEval、数学与代码旧能力测评继续沿原协议。

## 数据已冻结（01:15前）

Hausa训练6400条、输入2,596,327 token、assistant监督920,471 token，截断90条；开发256条。SW分别为2,884,423与1,078,347，因此只称样本/来源匹配。所有处理通过原官方LF tokenizer/assistant loss mask实现，不自行改训练框架。

许可证元数据：锁定Aya Collection及人工Aya、Global-MMLU源卡标记Apache-2.0；Belebele源卡标记CC-BY-SA-4.0。来源版本与元数据哈希已保存。详见[Collection源卡](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split)、[人工Aya源卡](https://huggingface.co/datasets/CohereLabs/aya_dataset)、[Global-MMLU源卡](https://huggingface.co/datasets/CohereLabs/Global-MMLU)、[Belebele源卡](https://huggingface.co/datasets/facebook/belebele)。

Collection的数字id在已审样本中等于本文件行号+1。SW6400与HA有448个数字ID碰撞、来源字段一致，但这些事实不足以认证跨语言语义配对；不能直接用它们建语义监督。

## 已冻结正式任务资产（02:18）

Hausa正式阅读来自原Meta官方ZIP中的hau_Latn.jsonl（900题），与现有EN/SW的ID/标签逐条一致。Global-MMLU ha固定revision0e619db、test14,042题/57科，dev285题，各科5个few-shot。CPU通过原Harness完整实例化58个具体任务并核验计数、答案与few-shot池；没有读取模型分数。Urdu缺Global-MMLU、HA/YO/UR缺当前BenchMAX官方配置的限制保留。正式任务不用于选择语言、LR、rank或候选。
