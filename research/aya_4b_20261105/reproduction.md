# Qwen3-4B SW SFT：复现说明

## 固定设置

模型为官方后训练 `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`。这里的 Base 指未做本轮适配的该模型，不是预训练 Base 变体。训练三组均为 seed42、6400条、1 epoch、400 optimizer steps、effective batch16（microbatch1×累积16）、assistant-only、non-thinking、BF16、2048、不 packing、AdamW、cosine、warmup0.05、无权重衰减、clip1。固定 Final400；不按英语下降提前停止或按正式测试选择。

FFT/SSU LR1e-5、ZeRO2 CPU optimizer offload；LoRA LR1e-4、rank16、alpha32、dropout0.05、七个 attention/MLP 投影。方法间超参数差异如实保留。只运行一个训练种子。原有其他实验及结果保持独立。

## 目录

- `configs/aya_4b_20261105/`：模型/数据版本、实际超参数、两套 Python 环境版本、冻结任务定义和生成执行配置。
- `scripts/aya_4b_20261105/`：数据重建、训练适配、固定评测、审计、统计和报告脚本。
- `results/aya_4b_20261105/reproduction-inputs-v1/`：公开源ID和输入/监督指纹，不含原文和 token IDs。
- `results/aya_4b_20261105/frozen_ssu/`：实际使用的 SSU 掩码，gzip 解压后与原文件 SHA256 完全一致。
- `results/aya_4b_20261105/analysis-v1/`：完成后写入的正式分数、逐题无文本统计和配对区间。

`source_snapshot_identity.json` 区分实际运行源码 SHA 与为移除服务器绝对路径而改写的公开源码 SHA。源码中的 `resolve()` 从显式环境变量取本机路径；不修改第三方框架或评分逻辑。配置里的 `${KEY}` 需先 materialize，不是 shell 命令。

## 重建训练输入

按 `reproduction_spec.json` 中的固定 URL/revision 获取四份公开源文件：Aya SW train、TyDiQA secondary_task train、XL-Sum SW archive、OPUS GlobalVoices v2018q4 en-sw zip。逐份验证 SHA256。不要另行抽样或打乱公开选择清单。

在训练环境运行：

```bash
python scripts/aya_4b_20261105/rebuild_selected_data.py \
  --identities results/aya_4b_20261105/reproduction-inputs-v1 \
  --model /your/models/Qwen3-4B \
  --aya /your/sources/aya-swahili-train.parquet \
  --tydiqa /your/sources/tydiqa-secondary-train.parquet \
  --xlsum /your/sources/xlsum-swahili_XLSum_v2.0.tar.bz2 \
  --globalvoices /your/sources/en-sw.txt.zip \
  --output /your/data/aya_4b/mixture-v1
```

必须通过6400条 input/label SHA校验，以及全部7个训练文件校验（含原始jsonl、来源清单和tokenized Arrow）。实际监督量为2,373,891输入token、324,892 shifted assistant tokens；与旧Aya配方的监督量不同。10条原生阅读上下文采用包含标注答案的确定性窗口；选中样本无SFT截断。阅读是短答案形式，不能把其训练损失当作Belebele选择题收益。

## 训练

配置环境变量 `MLCL_ROOT`、`AYA4B_MODEL`、`AYA4B_DATA`、`AYA4B_CHECKPOINTS`、`AYA4B_LOGS`，分别指向本仓库、模型、新数据根目录、全新输出根目录和日志根目录。`AYA4B_DATA` 下应存在 `mixture-v1/`。

先将 DeepSpeed JSON 和三份 YAML 分别用 `materialize_config.py SOURCE DESTINATION` 展开到独立工作目录；更新 YAML 的 `deepspeed` 指向展开后的 JSON。保留原始配置及展开后配置的 SHA256。不得覆盖已有输出。

训练入口为：

- LoRA：`train.py --config /your/materialized/lora.yaml`。
- FFT：单GPU `torchrun --nproc_per_node=1 --standalone train_cpu.py --config /your/materialized/fft_v2.yaml`。
- SSU：同 FFT，额外 `--mask /your/frozen-mask.json`。

FFT/SSU入口使用与实际运行相同的 DeepSpeed CPUAdam 绑定，并读取 `AYA4B_CPU_ADAM_SO` 指定的兼容扩展；在自己的训练环境构建或提供该扩展，勿复制不同 ABI 的二进制。原始实验因 PATH 下找不到 ninja 而复用已有相同环境扩展；失败日志和额外耗时已保留。

SSU 的 `--mask` 是普通 JSON **文件**。解压公开 `mask.json.gz`，核对 `identity.json` 中原文件 SHA。它保护1,816,796,928 / 4,022,468,096（45.1662%）唯一参数元素；不是全模型50%，也不是旧15%稀疏方案。英文Tulu校准500×2048块，包含数学偏向；该限制在解释效果时保留。掩码固定，不随本轮正式测试更新。

启动前核实共享服务器任务归属、显存和全账号GPU数；显式 `CUDA_VISIBLE_DEVICES=GPU-...`。通用选卡工具默认总计最多4卡；本轮专用开发入口记录临时5卡授权，不改变全局默认值。复现时须遵守自己服务器的当前授权。不随意终止他人进程。训练结束只导出固定Final权重，无 optimizer checkpoint；不能宣称从该导出精确恢复训练。

## 冻结正式评测

|任务|样本|形式 / 主指标|few-shot|执行|
|---|---:|---|---:|---|
|Belebele EN/SW|各900|四选项 likelihood / ACC|0|HF，batch4|
|Global-MMLU SW|14042，57科|四选项 likelihood / ACC|5|HF，batch1|
|MMLU EN|14042，57科|四选项 likelihood / ACC|5|HF，batch4|
|FLORES EN→SW / SW→EN|各1012|生成 / 语料级chrF++|3|vLLM V1，max new512|
|XL-Sum SW / EN|987 /11535|生成 / ROUGE-Lsum|0|vLLM V1，max new256|
|GSM8K|1319|生成 / 官方strict及flexible答案ACC|5|vLLM V0，max new1280|
|IFEval|541提示、834指令|生成 / 官方prompt及instruction strict|0|vLLM V0，max new1280|

生成全部greedy、non-thinking、seed42。MT同时报告BLEU13a；摘要保留ROUGE1/2。正式测试不参与训练、数据选择或Final选择。翻译始终分别报告两个方向，不能用一个方向替代另一个。

`materialize_tasks.py` 只展开三种数据/环境路径，保留 YAML 的 `!function` 和相对 include。Belebele来自锁定的Meta官方数据；MMLU/Global-MMLU任务YAML中保留Hub revision。GSM8K和IFEval原文件版本在路径、配置和身份清单中记录。

HF用 `run_fixed_hf.py --group ... --model ... --tokenizer ... --gpu-uuid ... --tasks /materialized/tasks --output /new/output`。LoRA额外 `--adapter /final/adapter`。完整权重传给 `--model`；LoRA的 `--model` 仍是官方Base。

MT/摘要准备沿用仓库 `scripts/bench_v2/download_data.py` 和 `prepare_datasets.py` 的官方源下载和预处理。`configs/aya_4b_20261105/generation_manifests/` 保存四份实际冻结 manifest 原始字节及身份清单；将对应 manifest 复制到各评测数据目录的 `manifest.json`，并逐一检查 `test.jsonl` 的 SHA256 与 `data_sha256`、条数和样本顺序一致。不要重新抽取 few-shot 示例或改写提示。manifest 中 `run_status=NOT_RUN` 是准备阶段字段，实际完成状态以每个模型的结果为准。`generate_v2_gpu_guard.py` 使用该manifest、模型、显式GPU和独立输出，固定 `--scope full --batch-size 8`；生成后用 `scripts/bench_v2/score.py` 评分。新版物理UUID断言只检查设备，不改题目、提示或输出。

内存配置v2统一用于四个模型：MT/摘要 maxseq8、prefill4096、显存比例0.65；GSM8K/IFEval maxseq8、batchtokens8192、显存比例0.65、eager。v1遇到LoRA长文本预填充OOM后保留为历史尝试；正式比较不混用v1。降低内存参数依据OOM，不依据分数。模型、参考、提示token SHA和样本顺序需跨模型完全相同。

## 结果重算与限制

完成后的无文本 `per_item_scores/` 保留逐样本分子分母与聚类身份；翻译 `corpus_statistics/` 保留逐句充分统计，重采样时重算语料级分数。无需训练模型、GPU或原始题目即可运行 `verify_public_results.py --analysis ... --output /new/receipt.json`，重算80项选定分数及60个配对区间。

区间为1000次、seed42的配对bootstrap，未进行多重比较修正，仅针对本轮固定seed42模型。Belebele按共享段落聚类，IFEval指令按提示聚类。不能解释成跨训练种子稳定性。

IFEval原官方实现对无效约束参数可能随机选择回退字母；独立审计验证原始官方flags、总数和确定性题目的重新评分，不伪称所有随机回退都可精确重放。主分数始终使用真实评测时保存的官方flags，不因重算差异修改结果。

原始逐题生成、训练日志、失败记录和模型权重保留在实验存储中；GitHub提供科学配置、代码、数据身份、训练统计、无文本逐题证据及图表。模型规模与数据同时改变，跨轮变化不能单独归因于模型规模或数据改进。

### GSM8K / IFEval命令模板

在独立vLLM环境和经核实空闲的单卡上，沿用仓库 `scripts/sft_v1/harness.py` 的 `run` 入口：

```bash
SFT_VLLM_TRACE=/new/core/completion-metadata.jsonl \
VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
python scripts/sft_v1/harness.py run \
  --model vllm \
  --model_args pretrained=/your/model,tokenizer=/your/Qwen3-4B,dtype=bfloat16,max_model_len=8192,gpu_memory_utilization=0.65,max_num_seqs=8,max_num_batched_tokens=8192,enable_prefix_caching=True,seed=42,enforce_eager=True,enable_thinking=False \
  --tasks sft_v1_gsm8k,sft_v1_ifeval \
  --include_path /your/materialized/tasks/core \
  --apply_chat_template --batch_size auto \
  --output_path /new/core/results --log_samples --seed 42
```

LoRA模型参数额外加 `lora_local_path=/your/final/adapter,max_lora_rank=16`。使用训练导出之外的模型、推理框架版本或解码设置时，结果应另立协议，不与本轮混合。

## 训练后增加的补充比较

XNLI SW/EN各5010题、XCOPA SW500题，统一0-shot、batch4、非思考、A/B/C字母likelihood ACC。该XNLI指令模板不同于Harness原始completion模板，分数不可直接等同。全部四个固定Final和Base均报告，不能按结果删除任务。开发smoke仅使用validation每任务2题。

`configs/aya_4b_20261105/supplemental/protocol.json`锁定源revision和文件SHA；其余YAML是实际任务定义的路径占位版本，真实原始YAML与公开版SHA单列于source_identity.json。用materialize_config.py展开`${DATA}`后运行HF评测。`prepare_supplemental.py`可从锁定parquet重建相同题目；其XNLI源缓存位置通过AYA4B_SECONDARY_ROOT配置，对应`datasets/unified-language-cl-20261104/sources/facebook--xnli/`，只需all_languages/test和validation两个parquet及含固定sha的metadata.json，不需下载train。XCOPA SW两个公开parquet由--xcopa指定。

补充结果可独立重算：`verify_public_results.py --profile supplemental --analysis results/aya_4b_20261105/supplemental-analysis-v1 --output /new/supplemental-proof.json`，完整门槛为12项分数和9个配对区间，不降低原正式表80项分数/60个区间的门槛。NLI按共享premise聚类，保存标签偏好诊断，避免把选择题字母偏好变化直接解释成纯语义推理学习。

翻译解码开发对照独立于正式表。固定64个FLORES dev句组、两方向、Base/LoRA、greedy与官方non-thinking采样两套预定义配置，全部512条生成的分数及语料统计保留；仅一个推理种子。详见translation_decoding_diagnosis.md。
