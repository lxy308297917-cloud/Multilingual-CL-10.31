# 使用与验证范围

## 依赖已有开源框架

历史训练：LLaMA-Factory 0.9.6.dev0，commit `47d407d2aabbd52e11d4f34c7a06b9de3e2f0a77`；torch 2.6.0+cu118、Transformers 4.57.6、PEFT 0.18.1。历史生成测评另用 torch 2.7.0+cu118、Transformers 4.53.2、vLLM 0.9.2+cu118；Harness 0.4.14.dev0，commit `d6de81643928d653435c431bae19945d41d32520`。两个环境分开，勿直接在同一环境盲目升级。

先使用已有环境，确认这些依赖及 PyYAML、safetensors 可用。入口自动加载本仓库 src，无需先安装本包。`pyproject.toml` 只定义包结构，不自动安装训练框架。

## 每个代码文件负责什么

| 文件 | 职责 |
|---|---|
| `scripts/train.py` | 参数检查、加载冻结资产、临时向 LLaMA-Factory 注入辅助损失 |
| `scripts/evaluate.py` | 转交上游 Harness CLI；没有自写 benchmark 评分器 |
| `scripts/analyze.py` | 从已发布逐题数值复算均分，不执行机制推理 |
| `src/s3_align/losses.py` | 英语表示保持、正文表示读取、固定映射对齐 |
| `src/s3_align/trainer.py` | 每个优化步一次英语保持及一次可选对齐；不改上游目标 SFT |
| `src/s3_align/assets.py` | 路径、SHA256、ID 顺序、教师形状和有限值检查 |
| `src/s3_align/subspace.py` | 历史 PCA128、ridge=0.1 的 CCA64 公式，输入须为中心化 CPU float64 |
| `src/s3_align/accounting.py` | 主训练样本、输入和回答监督词元计数、逐样本哈希记录 |
| `src/s3_align/harness.py` | 可选生成长度跟踪，评分仍调用 Harness |
| `src/s3_align/vllm_compat.py` | 历史 vLLM 0.9.2 的 GPU UUID/NVML 兼容补丁，非通用版本补丁 |
| `src/s3_align/__init__.py` | 包入口 |
| `tests/test_core.py` | 损失、梯度、读取位置、辅助暴露频次和 JSONL 回归 |

## 准备训练

1. 按归档协议准备 Bactrian-X 的 65,415 条 SW 数据及 LLaMA-Factory tokenized 数据；不能把示例路径下的任意新数据称为历史复现。
2. 将 `configs/train.yaml` 的模型、数据、输出路径改为部署路径。三组必须用不同的新输出目录；单 GPU、microbatch 2、梯度累积 8、seed42。这里只支持新训练，拒绝隐式恢复或覆盖。
3. 复制 `configs/assets.example.json` 到本机部署文件，填写文件路径与实测 SHA256：keep_contract/pool/cache/manifest 是历史 uniform36 英语保持契约及资产；alignment_pool/mappings/gradient_calibration 是冻结的 1,024 条配对池、映射及初始梯度标定。相对路径以部署 JSON 所在目录解析。T0 只需四项英语保持资产。
4. 先执行 CPU 预检：

```bash
python scripts/train.py --arm T0 --config configs/train.yaml \
  --assets configs/assets.local.json --check-only
```

预检不等于完整数据、上游环境或真实模型前向验证。新 GPU 运行仍需独立 smoke 和预算身份核验；不得据 CPU 对照声称严格复现。要复现报告，优先使用归档冻结入口及其原训练锁。

通过所需检查并显式选卡后，去掉 `--check-only` 才会训练；本次仓库整理没有执行该动作。T0 为英语保持；T1 增加统一坐标对齐；T2 增加双侧 CCA 对齐。保持损失 mu=1，映射权重从冻结标定读取。辅助梯度每优化步只计算一次，不再除以梯度累积次数；历史 keep_loss 包装器中的早期 microbatch 示例不代表最新 T0/T1/T2 调用方式。

整理版保留主训练预算记录，但尚未整合历史训练锁、初始 Adapter 哈希及训练结束全套身份断言，也不提供新的数据构造、教师缓存或梯度标定入口。这些冻结步骤仍在归档中，避免把未经 GPU 回归的新入口冒充已验证复现。

## 测评与分析

```bash
python scripts/evaluate.py --help
python -m unittest discover -s tests -v
```

测评参数遵循已安装 Harness；必须沿用对应实验冻结的任务配置、数据版本、聊天模板、few-shot、解码和评分。仅使用同名上游默认任务不足以声明与历史结果同协议。

默认不加载兼容补丁；只有设置 `SFT_VLLM_TRACE` 时才加载历史 vLLM 包装和长度跟踪。在更新的 vLLM 上先验证兼容性。大规模隐藏状态提取、干预及报告生成脚本保留在归档，不重复铺在当前 scripts。

报告目录中的 CSV/JSON 是研究结果，不是运行配置。历史报告内的服务器路径已脱敏，未公开的大型缓存、权重及原始文本不随仓库分发。
