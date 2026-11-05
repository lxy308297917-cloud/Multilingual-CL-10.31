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

本机原训练入口还显式设置了以下环境项，整理版独立复现时应记录同样的设置：

```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

每个推理任务使用独立的 `VLLM_CACHE_ROOT`，避免不同进程的编译缓存互相覆盖。
旧能力测评的 V0 引擎选择由 `evaluate_retention.py` 自行设置；不要用全局后端开关替换原生任务的执行设置。

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
   摘要用 `evaluate_summary.py`。MTOP 的准备、配置生成、训练和评测均设置 `UNIFIED_DATA_VERSION=prepared-v2`。
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

## Same-ID / Disjoint-ID复现

以下步骤沿用冻结的3200例/200步与同一ES Final，只在MASSIVE、XNLI、MTOP进行。
先完成全部主实验和预定开发中点。正式test不参与B队列、预算或模型选择。

1. 使用 `prepare_disjoint.py --track massive`、`prepare_disjoint.py --track xnli`，
   以及 `prepare_disjoint_mtop.py --track mtop`，从原始协议预留B ID重建TH缓存。
   这些命令必须使用训练环境，在CPU上工作。所有B文件写入独立`prepared-disjoint-v1`。
2. 运行 `make_p3_configs.py --tracks massive xnli mtop`。它从同轨主配置复制全部优化设置，
   仅改变B数据、缓存、输出和日志路径；复用相同的Base或ES Final，生成六份配置和身份文件。
   分析计划从仓库复制到本次生成配置目录，逐字节检查，不复制旧的成功标记。
3. 使用 `freeze_main_evaluation.py` 核验本次真实七轨道Base/Final的dev/test、通用测评及
   五轨道15个固定中点dev。工具验证模型权重、协议与输入资产、样本量和预测SHA；
   原生矩阵不齐时拒绝生成gate。其DONE是完成产物核验凭证，不是调度器进程历史。
4. MASSIVE/XNLI通过 `run_train_p3.py`，MTOP通过 `run_train_p3_mtop.py`，
   分别传入 `--config` 和 `--base-dev`。每次显式选择两张已检查可用的GPU UUID，账号总计最多4张。
   Base开发路径为 `TASK/base-dev-v1`，MTOP为 `mtop/base-dev-v2-syntax`。
5. 使用原来的 `evaluate.py` 对两个新增固定Final运行原生dev/test。
   输出名分别为 `disjoint_single_th-dev-v1`、`disjoint_single_th-test-v1`、
   `disjoint_seq_es_th-dev-v1`、`disjoint_seq_es_th-test-v1`；MTOP设置
   `UNIFIED_DATA_VERSION=prepared-v2`。评测读取原始A协议的同一dev/test，不读取B训练集当测试集。
6. `evaluate_retention.py` 的输出目录为
   `$MLCL_RESULTS/retention/TASK_disjoint_single_th/formal-v1` 或
   `$MLCL_RESULTS/retention/TASK_disjoint_seq_es_th/formal-v1`。
7. 对三个scope（`dev`、`test`、`retention`）分别调用
   `aggregate_p3.py --track massive|xnli --scope SCOPE`，或
   `aggregate_p3_mtop.py --track mtop --scope SCOPE`。
   汇总保留六模型、八个预定配对对比及95%条件区间，包含直接TH_A/TH_B对照。

固定中点输出名称是`ARM-dev100-v1`，SIB为`ARM-dev20-v1`。
`freeze_main_evaluation.py`要求MTOP、XNLI、XL-Sum、SIB-200、WikiLingua的这15项真实产物；
最先两轨的开发曲线仍单独保留。摘要中点直接使用正确v2评分器。
所有分类主实验的开发读出诊断需在`evaluate.py`传`--candidate-dev`，
适用于MASSIVE、XNLI、SIB-200，冻结相同SHA排序的64题，不据候选成绩更换正式指标。

## 整理版新增检查记录

六份P3配置已由整理版在CPU重新生成，与服务器预先冻结的六份配置逐字节一致，
必要身份字段、B数据/缓存SHA与顺序哈希也一致。未因此启动新GPU实验。
主评测gate在缺少实际WikiLingua矩阵时正确拒绝，未创建任何主完成或中点DONE标记；
完整正向gate按全部实际产物执行；其单独的CPU集成验证状态在整理版manifest中记录，不能把这个拒绝检查写成端到端复现完成。

完整CPU入口集成检查现已通过：使用原服务器已完成的七轨道Base/Final和15个中点产物，
重验模型权重文件、输入和预测哈希、样本量与协议后，在隔离目录生成新的核验凭证。
原始主实验完成标记与中点状态哈希保持一致。此检查没有重新训练或生成，不是独立GPU复测。

`analyze_p3_cohorts.py`只读取原先冻结的A/B train，汇总ID、规范化文本重复、分类标签或MTOP根意图计数及token预算。
它不根据这些统计调整样本；完整目标树和原文不写入公开汇总。对照训练与正式测评继续使用冻结版本。
