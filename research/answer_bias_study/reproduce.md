# 复现、数据用途与公开范围

本轮只做固定权重的机制诊断，不训练新模型。运行来源为上一轮已发布的 `75a7130`，公开版精选科学实现与汇总；完整原始数据、编码、模型、逐题预测、启动日志、路径和设备记录留在本地。不会发布服务器任务队列或资产/环境清单。

## 数据重建

`prepare.py`公开副本使用显式参数：`--tokenizer --knowledge-dev --reading-en --reading-sw --reading-ha --train-sw --train-ha --output`。知识来源目录准备官方锁定revision的`en-dev.parquet`、`sw-dev.parquet`、`ha-dev.parquet`。阅读为官方EN/SW/HA原始JSONL；使用历史冻结seed `ha-learning-reading-development-v1-20261103` 按整个段落选180题，再应用共同必要过滤，保留179题。源文件字段、SHA和语义用途见`configs/answer_bias_study/scientific_protocol.json`。

知识按学科固定SHA顺序分出1题校准、4题诊断，共同过滤后每语言53校准、219诊断。三语言必须共同保留/剔除语义ID。没有按模型分数调整题目。生成的items/content_free文件哈希应先核对；不同机器输入路径会改变protocol元数据哈希，应记录自己冻结的协议哈希并相应填入分析用科学配置，不能冒称原运行的元数据身份。

本轮阅读是已曝光test-origin补充诊断，不是独立dev或新的正式确认。知识官方dev亦有历史曝光。两者均不用于选择训练方法。知识先验估计从未读取阅读诊断题或正式知识test。原SFT数据只作必要精确问题重叠检查，没有开展新的全库审查。

## 固定模型评分

设置经过进程归属、可用显存和账号总占卡数复查的单张GPU UUID：

```bash
CUDA_VISIBLE_DEVICES=GPU_UUID CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
python scripts/answer_bias_study/evaluate.py \
  --model MODEL_DIRECTORY --data FROZEN_DATA_DIRECTORY \
  --output NEW_RESULT_DIRECTORY --label MODEL_LABEL
```

七个model label及官方底座revision见科学配置；训练Final均为400步。顺序模型拥有额外SW阶段，应分别解释其初始化和累计曝光。没有用任何新结果选checkpoint。

所有既有safetensors均为BF16部署导出（Base311张量，其余各310）；将其提升到同一个FP32/SDPA推理路径，不换用训练FP32主值、不消除导出时的量化误差。使用非思考模板、batch4与答案位置FP32投影。`core.py`对正确移位位置计算完整候选续写总似然；异长内容保留token均分和总分。每个真实模型先在独立校准题上对照完整原生前向，逐模型误差见validation.json。容差是工程数值校验，不能担保所有近并列题的argmax跨实现一致；原字母LP差≤1e-3与平均概率差≤1e-6已单列。

环境沿用项目已有训练环境与已公开栈锁，未安装或修改共享环境。当前执行栈为PyTorch2.6.0+cu118、Transformers4.57.6、NumPy2.4.6；不能用vLLM/BF16正式分数替换本轮FP32诊断分数。

## 分析

公开版 `analyze.py` 使用 `--protocol configs/answer_bias_study/scientific_protocol.json --run-root RESULTS_BY_MODEL --output NEW_ANALYSIS_DIRECTORY`。RESULTS_BY_MODEL下按model label组织每组`DONE.json`、`predictions.jsonl`、`content_free_predictions.json`。它们包含未分发的逐题本地证据；只下载公开聚合JSON不足以重算配对区间。

`write_reports.py --analysis ANALYSIS_JSON --output REPORT_DIRECTORY` 生成完整结果表与跨语言组合；`plot_results.py`同样接受`--analysis --output`，生成PNG。只读原始产物，不更改正式benchmark结果。为新运行指定独立输出目录。

CPU验证：

```bash
PYTHONPATH=scripts/answer_bias_study \
python -m unittest discover -s tests/answer_bias_study -v
```

六项覆盖完整因果似然oracle、多token与padding、共同前缀、轮换回映射和PriDe分解、过滤、段落加权区间与纯位置模型的均匀并列反例。核心/evaluator/test与实际执行副本逐字一致；公开prepare/analyze仅改数据入口参数及历史180题的可重建选择，原稿与公开源码哈希分别记录。

## 解释边界

校准与位置平均是已有诊断基线，不是My Method。去掉字母改变了任务提示，token均分也不是纯语义能力测量。跨语言同内容一致可能共同答错，必须同时看正确性组合。所有区间为固定checkpoint及固定53题校准先验下按诊断题/段落抽样，不包含校准集合抽样不确定性，也不代表训练种子重复性。正式0/5-shot、BF16与当前FP32/0-shot任务分开，不跨协议直接相减。缺少新独立平行阅读确认集与权重干预，不能据此宣布因果定位或训练方法有效。
