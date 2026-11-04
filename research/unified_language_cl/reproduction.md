# 统一语言顺序实验：复现入口

本页对应 GitHub 整理版，建议在独立检出目录复现。服务器原执行源码及版本文件保持不变，
不要将这里的同名整理脚本直接覆盖到仍有实验运行的目录。

## 状态与范围

七个独立任务轨道；每个轨道分别执行 Base→ES、Base→TH、同轨道 ES Final→TH。
FR 仅评测。Base 指官方 Qwen3-1.7B（revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`），不是另一个同名预训练 checkpoint。
正式结果来自报告列出的实际执行代码。这里将路径集中到 `runtime.py`，合并版本文件名；
`portability_manifest.json` 对应原执行源码与整理后源码的 SHA256。
整理版已完成 CPU 校验，尚未重新完成一次 GPU 端到端复现，不能把它称为第二次独立复测。

## 环境与路径

使用独立训练和推理环境，沿用项目锁定的 LLaMA-Factory、Harness、vLLM 与模型版本。
不要把 vLLM 环境中的 Torch/Transformers 覆盖到训练环境。没有安装依赖、修改系统 Python
或重写共享环境的自动步骤。`runtime.py` 通过以下环境变量接受实际部署路径：

```bash
export MLCL_PROJECT_ROOT=/path/to/Multilingual-CL-10.31
export MLCL_TRAIN_PYTHON=/path/to/train-env/bin/python
export MLCL_EVAL_PYTHON=/path/to/eval-env/bin/python
export MLCL_MODEL=/path/to/Qwen3-1.7B
export MLCL_DATA=/path/to/data/unified_language_cl
export MLCL_CHECKPOINTS=/path/to/new/checkpoints
export MLCL_LOGS=/path/to/new/logs
export MLCL_RESULTS=/path/to/new/results
export MLCL_CONFIGS=/path/to/new/generated_configs
```

未设置的输出目录默认位于仓库 `runs/reproduction/`。每次复测选用新目录；脚本拒绝覆盖已有训练与测评。
基础模型由既有 `scripts/verify_model_lock.py` 验证，不重复下载已有模型。
训练和推理命令须显式设置 `CUDA_VISIBLE_DEVICES` 为经检查的空闲 GPU UUID；
本项目执行时账号总占卡上限为4张，双卡训练也计入其中。

摘要使用隔离的作者 ROUGE 源码目录 `MLCL_ROUGE_SOURCE`，以及
`MLCL_METRICS_ADDONS` 指向的独立附加包目录（pythainlp 5.1.2、pyonmttok 1.37.1）。
NLTK stopwords 放在 `$MLCL_DATA/metrics_resources/nltk_data/corpora/stopwords.zip`，
SHA256 为 `48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b`。
旧能力测评另设 `MLCL_OLD_NLTK`，提前配置 punkt/punkt_tab 等所需离线资源，避免任务启动时联网下载。
IFEval 推理环境须在解释器启动时设置 `langdetect.DetectorFactory.seed=42`，与本次实际环境一致；
`setup_retention.py` 会检查该设置，不会替用户修改环境。

## 数据准备与冻结

在仓库根目录，以下命令仅准备数据或执行 CPU 检查，不启动训练：

```bash
"$MLCL_TRAIN_PYTHON" scripts/unified_language_cl/fetch_sources.py
"$MLCL_TRAIN_PYTHON" scripts/unified_language_cl/fetch_metrics.py
for task in massive mathoctopus xnli sib200; do
  "$MLCL_TRAIN_PYTHON" scripts/unified_language_cl/prepare.py --track "$task"
done
"$MLCL_TRAIN_PYTHON" scripts/unified_language_cl/prepare.py --track mtop --data-version prepared-v2
"$MLCL_TRAIN_PYTHON" scripts/unified_language_cl/audit_raw.py --track xlsum
"$MLCL_TRAIN_PYTHON" scripts/unified_language_cl/prepare_summary.py
"$MLCL_TRAIN_PYTHON" scripts/unified_language_cl/audit_wikilingua.py
"$MLCL_TRAIN_PYTHON" scripts/unified_language_cl/prepare_wikilingua.py
"$MLCL_TRAIN_PYTHON" scripts/unified_language_cl/make_wiki_clusters.py
"$MLCL_TRAIN_PYTHON" scripts/unified_language_cl/freeze_gate.py
"$MLCL_TRAIN_PYTHON" scripts/unified_language_cl/setup_retention.py
```

`sources.lock.json` 固定42个公共资产的来源、版本、大小和 SHA256。
下载器验证完整文件，保留已存在但不匹配的文件并报错；不执行远程数据集加载代码。
WikiLingua 使用受限 primitive pickle 解析器及明确的 tar 成员名单。
下载和重建得到的原始文本、模型、optimizer state 不放入 GitHub。
相关数据许可证和 WikiLingua 正式测试映射子集限制见数据报告。

`freeze_gate.py` 根据重新生成的文件验证七轨道及全部28个训练缓存的样本数、token 总量、
assistant mask 和模型/tokenizer 身份，然后生成新的 P0 gate。
这属于输入一致性验证，不是对语言质量或语义对齐的独立人工认证。

## 训练与测评顺序

1. 每轨道先运行固定 Base 开发评测。分类/推理/数学用 `evaluate.py`；
   摘要用 `evaluate_summary.py`。MTOP 的准备和评测均设置 `UNIFIED_DATA_VERSION=prepared-v2`。
2. 运行 `make_configs.py --track TASK` 生成独立三组配置。摘要添加 `--micro 1`；
   其他轨道使用 `--micro 2`。有效 batch 均为16。
3. 首次使用本机环境时，用 MathOctopus 的 `--engineering --micro 2` 配置执行
   `torch.distributed.run --nproc_per_node=2 --standalone scripts/sft_v1/train_formal.py
   --config CONFIG --stop-at 2`，再执行 `validate.py --config CONFIG --steps 2`。
   验证产物位于独立 engineering 目录；它不是正式实验结果。
4. 正式段通过 `run_train.py --config CONFIG --base-dev BASE_DEV_DIRECTORY` 启动。
   每段200 optimizer steps/3200例，SIB为40步/640例。
   SEQ 只读取同轨道 ES Final 的权重，重置 AdamW 和 scheduler。
5. 对 Base 和三个固定 Final 运行相同 ES/TH/FR/EN 开发及正式测试协议，
   对固定 Final 运行 `evaluate_retention.py` 的 EN Belebele、GSM8K 和 IFEval。
   固定中点 checkpoint 的评测仅用于开发学习曲线，不选择新的 Final。
6. 使用 `aggregate.py`、`aggregate_retention.py`、`report_curves.py` 汇总。
   正式测试不用于学习率、预算、数据或 checkpoint 筛选；不因 EN 下降早停。

摘要整理版已经直接使用纠正后的多语言句子 tokenizer；历史执行记录采用
“原始生成不变＋统一 CPU 重评分”，二者版本与来源分别记录。
ROUGE-1/2 回归、Thai 非同一文本部分匹配和分句重排 oracle 见评分修正文档。
旧摘要分数保留，不能作为主结果混入新表。

## 已验证内容与剩余限制

- 整理版5轨道×4语言×2样本，共40例提示、实际 LF 输入 token 和监督 mask 与原缓存一致。
- 多语言 ROUGE 的16个独立 oracle 通过；9个作者源码文件从固定 commit 重新下载后逐文件 SHA 验证。
- 三个 EN 任务在训练和推理环境中递归展开后的提示、过滤器、函数源码、解码配置及本机数据路径一致。
- 七轨道28个训练缓存的完整样本数、mask、输入和监督 token 总量验证通过。
- 上述 CPU 验证不证明不同 GPU/依赖版本具有逐 bit 相同的训练结果；未执行的 GPU 复测须标记 NOT_RUN。
