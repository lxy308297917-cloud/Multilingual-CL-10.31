# 复现与公开范围

本轮采用既有LLaMA-Factory/Transformers/DeepSpeed栈，以进程内Trainer子类加入一个辅助损失；不修改共享安装，不新建第二套训练框架。所有数字使用固定100步终点。公开科学实现、配置与汇总；完整原始输入、编码、逐题预测、模型/原生优化器状态和服务器日志留在本地。

## 1. 数据

从[Uhura ARC-Easy](https://huggingface.co/datasets/masakhane/uhura-arc-easy/tree/561efa15460cba0965212b77f09bf6a7f57939a9)锁定revision下载`en_train.json`、`sw_train.json`、`ha_train.json`。其[原论文](https://arxiv.org/abs/2412.00948)说明了专业人工翻译来源。只用原train，不读取其官方test。

准备原项目的SW/HA6400 SFT文件、原MMLU EN test/dev parquet和Belebele EN JSONL，按[科学配置](../../configs/decision_stability/scientific_protocol.json)校验输入SHA，然后运行：

```bash
python scripts/decision_stability/make_data.py \
  --source UHURA_LOCKED_DIRECTORY --project REPOSITORY \
  --output NEW_FROZEN_DATA_DIRECTORY --tokenizer QWEN3_TOKENIZER \
  --ha-train HA6400_JSONL --sw-train SW6400_JSONL \
  --previous-mmlu ORIGINAL_MMLU_TEST_PARQUET \
  --previous-dev ORIGINAL_MMLU_DEV_PARQUET \
  --previous-reading ORIGINAL_BELEBELE_EN_JSONL
```

语义ID以固定SHA顺序分出160/120/120，三个语言共同分组。四个编码/划分文件应匹配科学配置中的SHA；重建时间戳会改变protocol.json元数据身份，须记录自己的元数据SHA，不冒称旧运行身份。公开副本已完整重建400个ID，四个编码文件逐字节一致。数据不是全新采集题，也不保证基础模型未见于预训练；不重新声称做过完整母语者逐题认证。

## 2. 教师和独立读出

W1是原SW Aya6400全参数SFT400的固定导出，官方底座是后训练Qwen3-1.7B revision`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。教师只处理辅助160题中的EN/SW，53个ID满足两语言共同可靠性门槛；按同固定顺序供B/C/D使用，100步实际覆盖前50个ID的EN和SW。

```bash
CUDA_VISIBLE_DEVICES=GPU_UUID python scripts/decision_stability/score.py \
  --project REPOSITORY --model SW_FFT_FINAL400 \
  --data FROZEN_DATA_DIRECTORY --output NEW_TEACHER_DIRECTORY --mode teacher
```

Teacher是原生BF16前向+四候选FP32归一化，学生为训练态ZeRO3，微小数值差保留，不用后见的“teacher完全相同”断言掩盖。训练态首步内容KL约0.0006。

开发或留出评分将`--mode`设为`dev`或`holdout`，使用固定FP32提升导出权重的似然路径；读出与旧正式BF16分数分表。所有四组训练完成前禁止评分新holdout。P0仅W1和原A100查看dev。原字母、四轮平均、完整内容raw/token-mean及16交叉标签/位置控制均保留，不选择最有利prompt。

## 3. 固定100步训练

公共配置`configs/decision_stability/pilot100.yaml`仅将本机路径映射为`/local/...`角色占位符，按实际路径填写并为每组使用独立output/logging目录。配置仍是6400样本/一epoch/400步cosine与20步warmup，由callback停止于100；不能把max_steps改成100，导致调度曲线改变。

两GPU、micro1、累积8、effective16；seed42、AdamW1e−5、WD0、clip1、BF16、SDPA、2048、assistant-only、non-thinking、不packing。沿用已有环境：训练PyTorch2.6.0+cu118、Transformers4.57.6、LLaMA-Factory0.9.6.dev0（既有47d407d2来源锁）、DeepSpeed0.17.6。不要改系统或共享环境。

```bash
CUDA_VISIBLE_DEVICES=GPU_UUID_0,GPU_UUID_1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 \
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/decision_stability/train_pilot.py \
  --project REPOSITORY --config METHOD_CONFIG_YAML \
  --data FROZEN_DATA_DIRECTORY --teacher TEACHER_DIRECTORY/teacher.json \
  --mode content --stop-at 100
```

B将mode改为`marginal`。D使用**`train_replay.py --mode replay`**，实施完整词表上的正确答案末token CE；`train_pilot.py`的候选四类CE仅是参考实现，未用于D正式结果。共同聊天空白前缀作为上下文，不额外监督EOS。三组各四次旧任务前向/更新、同题同轮换同lambda1；附加loss一更新只加入一次，按实际HF累积归一化条件处理，记录运行时factor；本轮`model_accepts_loss_kwargs=true`且num_items存在，factor=1。没有把auxiliary forward算进HA样本/监督token计数。

A复用原顺序FFT的checkpoint100，经训练参数、400步调度器、source和两rank各800条实际输入/标签曝光核验；不是新训练种子。B/C/D与A的HA曝光全字段对比，原生checkpoint和Adam状态保存。工程探针与科学结果目录分开。

B是每题四旋转的学生标签均值拟合教师按语言全局标签边际，属于minibatch边际控制；不是全cohort严格约束。C拟合内容ID对齐后的教师/学生四旋转概率均值。D是同新辅助题的旧语言监督rehearsal，不是另外取Aya SW训练样本混入。D的任务均衡aux CE权重为1，未按全体新旧token重新归一化。三种loss的梯度量级不强行宣称匹配；单lambda pilot不代表调优后的算法排名。

## 4. 正式benchmark和统计

在固定100 checkpoint上沿用已有评测代码：`scripts/run_checkpoint_eval_17b.py`的`build_command`生成Belebele/EN MMLU/SW Global-MMLU命令；HA采用既有冻结HA任务定义（900阅读0-shot、14042知识5-shot）。核心生成使用`configs/sft_v1/eval_tasks`和`scripts/sft_v1/harness.py`，vLLM V0、BF16、greedy、seed42，保留GSM strict/flexible、IFEval提示/指令strict/loose及SW相应任务。正式原始逐题输出通过既有accuracy-audit重新计分。

翻译采用`bench_v2/generate.py --scope full --batch-size32 --method full`及`bench_v2/score.py`，双向FLORES1012，vLLM V1；同3-shot、固定Base tokenizer、non-thinking、max_new_tokens512。和历史W1 full-v2逐题prompt-token SHA对齐，不混用旧500句HF结果。沿用vLLM0.9.2+cu118/Transformers4.53.2环境。任务/数据锁保留在既有公共代码与各次本机原始记录中；公开交付不包含服务器排队启动器。

`analyze.py --project REPOSITORY --inputs MODEL_TO_RESULT_DIRECTORY_JSON --output NEW_DIRECTORY`分析新平行题。`analyze_benchmarks.py --inputs FORMAL_INPUTS_JSON --output NEW_DIRECTORY`读取逐题harness文件与翻译结果，严格检查相同doc、prompt及ID后进行配对统计。配对文件格式：每模型`results`是原生`results_*.json`路径列表；`translation`是两个方向到该模型生成目录的映射。要重算区间必须本地重跑或拥有原始逐题结果，公开聚合JSON不能恢复未分发的逐题证据。

新平行题4000次语义IDbootstrap；正式ACC2000次，阅读按原段落、IFEval按prompt保留指令数量聚合，其余按原题。翻译1000次配对corpus chrF++，缓存并相加sacrebleu充分统计量，和完整corpus scorer及含重复的重采样oracle对照，不平均句级分数。全部seed42、未多重校正、单训练种子；不把区间跨0当作等价证明。`plot_results.py`从聚合JSON生成独立PNG。

正式test有历史曝光，只作固定budget的补充比较，不参与teacher过滤、步数/lambda调整或checkpoint选择；新的holdout才是本项目本阶段未触碰的留出集。不因EN下降早停、不根据结果自动扩到400步。

## 5. 验证

```bash
PYTHONPATH=scripts/decision_stability \
python -m unittest discover -s tests/decision_stability -p 'test_*.py'
```

六项CPU检查覆盖内容ID映射/梯度、边际与语义约束区别、梯度累积，以及全词表Replay对非候选token的梯度。另有双卡原生两步预检和实际曝光对齐记录；它们是工程验证，不作为方法收益证据。训练代码和两步探针原始来源哈希保留，报告区分实际执行版本与公开路径占位映射。
