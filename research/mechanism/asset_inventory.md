# 已验证资产清单

检查时间：2026-11-02T16:12:49.094190+08:00

## 模型

Qwen/Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。本机权重可读；官方后训练模型，28层，hidden2048，tied embedding/head。路径及文件大小在asset_inventory.json。基线结果复核80指标与28原始结果文件一致；该核验不等于新的机制结果。

| 方法 | 保存格式 | Final权重可读 | 100/200/300/400 |
|---|---|---|---|
| full | HF full safetensors | True | 有/有/有/有 |
| lora | PEFT LoRA adapter | True | 有/有/有/有 |
| ssu | HF full safetensors | True | 有/有/有/有 |

LoRA Final只有adapter，没有已保存的merged完整模型。需从同revision Base通过PEFT加载并按实际scaling合并，不能用A/B矩阵直接与Full相减。Full/SSU均为完整模型；保存dtype可能混合，分析加载统一显式float32并披露舍入区别。

## 模板与数据

训练为Qwen3非思考chat、assistant-only、cutoff2048；复用固定Base tokenizer。三个Final的tokenizer.json均与Base相同，但tokenizer_config.json不同，不能直接混用模板；机制探针统一使用Base tokenizer.apply_chat_template(enable_thinking=False)并记录prompt哈希。

Aya训练6400条，EN/SW dev各256条。EN/SW dev不是可靠的一一平行集，不能按行配对。开发数据可读取，但不等价于覆盖数学、指令和阅读的可信benchmark。独立probe的schema、来源、质量状态和类别覆盖仍需构建；正式测试和演示不用于挑层。

现有Harness、vLLM、BenchMAX正式任务代码可追溯，但机制probe用独立入口，不重复改写正式评分。参数漂移审计无需GPU；真实干预smoke尚未运行。

## 已取消资产

paired/source/shuffled/replay四组于16:04收到明确停训指令并完成退出核验；checkpoint分别保留至200/200/200/300，不是400步Final，不作为本次机制实验的baseline。协调器已退出，四个后续测评队列inactive。

## 声明与证据边界

本清单现场检查了目录、权重文件大小、checkpoint中权重文件、tokenizer哈希和数据行数，未再次逐字节哈希所有大模型文件。Base revision来自已冻结模型锁；首次真实加载会进一步检查结构/shape/dtype/tied别名。不能将配置声明当作加载、干预或功能分离已验证。
