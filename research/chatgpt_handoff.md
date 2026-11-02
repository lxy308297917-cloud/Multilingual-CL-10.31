# 给ChatGPT的项目交接：寻找my method

## 现在需要解决什么

用户最新要求：不要继续消耗时间做许多训练种子；重点是找到一个有创新性的CL + 多语言适配方法，并完成与同类型成熟baseline的比较。已有完整seed42结果足以作为现象和工程起点。**my method尚未实现、尚无分数。** 本仓库不将现有SSU或英语保护改名为创新。

请先帮助明确：方法要解决的具体失败模式、与已有工作的实质区别、可证伪的机制假设、最小实现路径、对应baseline以及必要消融。创新性需要文献核查，当前未完成新方法的文献排重。

## 已有研究现象

Qwen3-1.7B在Aya SW6400条指令上训练一轮后，SW阅读与知识ACC提高，部分通用任务（IFEval、GSM8K、SW→EN翻译）下降。英语阅读和选择题知识提高，不能将所有英语能力统一称为遗忘。SSU比FFT保留更多指令/数学表现，同时仍有目标阅读收益。完整上升/下降/持平结果见[结果总览](../results/README.md)，不只选符合叙事的指标。

## 当前主线的固定设置

- 原模型：`Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，发布的后训练模型；1,720,574,976个唯一参数。
- 数据：Aya Collection SW固定6400条；一轮400步；输入2,884,423、移位assistant监督1,078,347 token；有效batch16。
- 非思考chat模板、assistant-only、BF16、cutoff2048、不packing、cosine、warmup5%，主要比较固定预算Final。
- FFT：LR1e-5、双卡ZeRO3，micro1×accum8×world2。
- LoRA：r16/alpha32/dropout0.05、七种Attention/MLP投影，LR1e-4，单卡micro2×accum8。
- SSU-Wanda：沿用锁定的Tulu500英语校准掩码，保护704,737,792/1,720,574,976≈40.9594%唯一参数；LR1e-5、WD0、单卡ZeRO2 CPU optimizer offload。**不是历史SSU15%。** Tulu500数学占比较高，需考虑校准代表性。
- 训练工具：LLaMA-Factory；评测：lm-evaluation-harness、vLLM、BenchMAX及官方评分。版本在`configs/sft_v1/stack.lock.json`。

## 应如何设计my method与baseline

1. 优先只做SW、单个开发种子；先证明机制与正式任务收益，再考虑其他语言/多seed。
2. 根据方法所属类型选择成熟对应baseline。稀疏方法匹配实际更新比例与候选参数范围；重放方法匹配英语样本/token及计算开销；投影方法使用同样可用的校准/记忆数据。FFT/LoRA是通用参考，不自动构成任何方法族的充分对照。
3. 固定同一初始化、目标数据、输入顺序/曝光、预算和评测。英语校准/重放额外token、额外前后向和显存/耗时单独报告，不能隐去。
4. 用独立开发材料检验假设和选择配置。正式测试已经用于报告和错误分析，不能再当作未见开发集反复调参；后续比较明确标记曝光情况。
5. 不因英语下降早停普通baseline；保留Final及实际失败，不人为制造收益/遗忘。
6. 从成熟开源实现/扩展接口作最小集成，不再维护多套自写Trainer或评分平台。不能把三处掩码抽查误写为全部参数的逐元素证明。

## 固定主要任务与诊断限制

- SW/EN Belebele各900题：ACC；EN MMLU/SW Global-MMLU各14042题：ACC。
- FLORES双向各500对：3-shot、SacreBLEU2.6.0语料级chrF++。
- EN IFEval：541提示/834指令，0-shot，提示级/指令级strict为主。
- GSM8K：1319题5-shot，strict最终答案ACC为主，flexible仅辅助。
- 原生补充：ProX EN/SW各11759题5-shot生成EM、SW MGSM250题8-shot、SW BenchMAX429提示/633指令0-shot。
- 原生SW ProX Base大量空输出；选择题字母偏置、数学提取标记、翻译连续重复均已记录。保持官方分数，另列诊断，不将代理指标或修改后的评分冒充主要收益。

## 哪些工作已完成，哪些已停止

- seed42 Base/FFT/LoRA/SSU完整七组测评，共80个选定官方指标单元；各模型核心32叶任务/26057生成请求。
- 已完成的seed43/44任务保留在141项快照。seed43 FFT/SSU全套完成；其余复测按用户最新要求停止，不能把不完整训练或测评冒充Final。
- 旧MURI Full正确LR接续r3已完成1250步一轮及六组原HF测评；旧r2偏差结果、LoRA1250一轮/2500两轮分开。
- 旧Qwen2.5-1.5B D2、旧A线MADLAD CPT40M、新Aya SFT的数据/预算/方法/协议不同，不能把差异全部归因于模型更换。
- 当前不运行额外GPU任务。恢复已主动停止的复测需要后续明确指令。

## 代码阅读入口

- `scripts/sft_v1/train_formal.py`：成熟训练框架的正式Full/LoRA入口。
- `scripts/sft_v1/train_ssu.py`、`scripts/ssu_mask_adapter.py`：SSU薄适配。
- `scripts/sft_v1/train_resume_safe.py`：已验证的单卡CPUAdam恢复绑定；旧canonical SSU resume修复尚未合并，当前请用该明确入口。
- `configs/sft_v1/formal_sw_*_seed42.yaml`：共同预算及方法配置。
- `scripts/sft_v1/harness.py`、`configs/sft_v1/eval_tasks/`：官方任务与实际生成观察。
- `scripts/sft_v1/export_baselines.py`：已完成官方结果导出；各种`analyze_*.py`仅分析保存输出。
- `research/sft_v1.md`：按时间保留的研究记录；历史“正在运行/下一步”等只代表当时快照，以`STATUS.md`为准。

## 工程与已知问题

模型实体在`/path/to/storage/checkpoints/`，统一入口为`/path/to/workspace/models/checkpoint/`；数据/依赖环境在服务器外部目录，仓库不打包第三方源码、数据集或权重。跨机器复现需按锁文件准备资产并调整本地绝对路径；当前不是一键可移植发行包。

旧CPUAdam从checkpoint恢复曾发生scheduler和实际optimizer.param_groups引用失配，已在`train_resume_safe.py`修正并验证模型/FP32 Adam/RNG/实际LR恢复；BF16最终权重不承诺逐位相同。本轮seed42三方法均从头训练，不受该恢复故障影响。双卡Full必须使用`queue_full.sh`的`expandable_segments:True`，通用入口曾在seed43四步后OOM；失败独立保留，修正后seed43完成。

多人共享服务器，账号最多四张GPU，显式CUDA UUID；不操作他人进程。检查data0和data1真实路径/余量，日志与产物独立存放。用户要求至少持续工作到2026-11-03中午12点，现重心是方法研究，不再扩展训练种子。


## 公开发布范围

这是精选公开副本：服务器运行管理文件、AGENTS、完整资产清单、进程状态和内部运行记录不包含。源代码中的本地路径统一替换为`/path/to/...`，算法逻辑保留。当前不是开箱即用发行包；需准备锁定上游、模型与数据并填写本地路径。CSV/JSON保留完整官方数值与原始来源哈希，不包含服务器路径。
