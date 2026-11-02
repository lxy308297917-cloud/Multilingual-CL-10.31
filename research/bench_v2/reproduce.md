# Benchmark v2 复现

运行前读取[固定协议](benchmark_protocol.md)和[限制](limitations.md)。以下命令在仓库根目录执行；路径由调用者配置，模型必须是对应冻结Final，不能用中间checkpoint代替。共享GPU先检查显存、利用率和进程归属，显式选择空闲GPU UUID；不要修改他人进程。

## 依赖和数据

推理环境是Python3.11、torch2.7.0+cu118、vLLM0.9.2+cu118、transformers4.53.2。指标为SacreBLEU2.6.0、rouge-score0.1.2、langdetect1.0.9。完整版本清单和关键wheel SHA见`configs/bench_v2/environments/`。vLLM的CUDA变体必须与实际驱动及torch匹配，不能把默认PyPI wheel当作本次cu118 wheel。

代码静态处理单独使用EvalPlus0.3.1环境；已实际兼容的依赖包括numpy1.26.4、pyarrow17.0.0、pandas2.2.3、datasets4.0.0、huggingface-hub0.36.2。它不加载或执行生成函数。

```bash
export EVAL_PY=/absolute/path/to/evaluation-env/bin/python
export CODE_PY=/absolute/path/to/code-static-env/bin/python
export BENCH_DATA=/absolute/path/to/bench-v2-data
export FLORES_DIR=/absolute/path/to/flores200_dataset
export BASE_MODEL=/absolute/path/to/Qwen3-1.7B
export NLTK_DATA=/absolute/path/to/pinned-nltk-data

"$EVAL_PY" scripts/bench_v2/download_data.py \
  --lock configs/bench_v2/downloads.lock.json --output "$BENCH_DATA/raw"
"$EVAL_PY" scripts/bench_v2/prepare_datasets.py \
  --raw "$BENCH_DATA/raw" --flores "$FLORES_DIR" \
  --legacy-config configs/flores_sw_en_500_seed42.json --output "$BENCH_DATA/prepared"
"$CODE_PY" scripts/bench_v2/prepare_code.py \
  --raw "$BENCH_DATA/raw" --output "$BENCH_DATA/prepared"
"$EVAL_PY" -m unittest discover -s tests/bench_v2 -p 'test_*.py'
```

FLORES复用[官方原版FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200)；归档SHA应为`b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`。不要换用较新FLORES+后沿用旧ID和哈希。需要dev、devtest及其metadata原文件；数据准备只读取指定test成员，不执行远程dataset脚本。NLTK Punkt/Punkt_tab需在开始前准备，实际资源文件校验和在环境证据中记录。

数据下载器先验证已有文件，校验不符即报错；不会覆盖错误文件。中断下载保留`.part`供检查，不盲目重试。`--verify-only`只校验现有数据。

## 单模型单任务

```bash
export CUDA_VISIBLE_DEVICES=GPU-replace-with-verified-free-uuid
"$EVAL_PY" scripts/bench_v2/generate.py \
  --dataset "$BENCH_DATA/prepared/xlsum_sw_v2" \
  --model "$BASE_MODEL" --tokenizer "$BASE_MODEL" --method base \
  --gpu-uuid "$CUDA_VISIBLE_DEVICES" --scope full \
  --output /absolute/path/to/new-run
"$EVAL_PY" scripts/bench_v2/score.py \
  --run /absolute/path/to/new-run --output /absolute/path/to/new-run/score-v1
```

`--scope smoke/pilot/full`分别对应固定4条、固定128条及完整任务。只对摘要运行Pilot主流程，Full结果与Pilot分开。`--batch-size`默认32，本轮全部使用32。普通新任务默认账号最多4张GPU；本轮用户明确允许所有空闲卡，所以实际一次性启动传入`--max-user-gpus 8`，未修改全局默认。

FFT/SSU分别将`--model`设为对应Final路径，`--method full/ssu`；tokenizer仍用Base。LoRA使用Base `--model`、`--method lora --adapter /absolute/path/to/final-adapter`，不预合并、不退回Base。manifest包含Base与Adapter实体文件SHA。

任务目录：`flores200_en_sw_full_devtest_v2`、`flores200_sw_en_full_devtest_v2`、`xlsum_sw_v2`、`xlsum_en_v2`、`humaneval_plus_v2`、`mbpp_plus_v2`。

## 代码生成与静态检查

生成仍由相同`generate.py`完成，不执行生成的代码。之后：

```bash
"$CODE_PY" scripts/bench_v2/analyze_code.py \
  --run /absolute/path/to/code-run \
  --official-data "$BENCH_DATA/raw/HumanEvalPlus-v0.1.10.jsonl" \
  --output /absolute/path/to/code-run/static-v1
```

MBPP使用`MbppPlus-v0.2.0.jsonl`。输出标准`{task_id, solution}` JSONL，可转交有可靠隔离的机器评测。本次不提供绕过隔离的执行入口；所有pass@1维持NOT_RUN。执行机器应独立确认最小只读挂载、非root、禁网络、内存/CPU/pids限制和每题timeout实际生效，使用同版本EvalPlus；普通venv、subprocess和timeout不能作为沙箱。

## 断点与汇总

恢复需相同源代码commit及核心脚本哈希、同一manifest、模型实体文件、数据、提示和参数。原输出目录传`--resume`，跳过已完成ID；损坏JSONL、冲突或重复不静默忽略。如主仓库HEAD已前进，应在独立检出恢复原记录commit，不改运行中的代码。故障日志和旧产物保留，修复后独立目录并记录原因。

可以用公开的`index_runs.py --run /path/to/first-run --run /path/to/second-run --output /path/to/new-index.json`从明确选择的已完成run重建分析索引，无需私有队列。每个run须有DONE及唯一的score-*/static-*评分目录；重复同方法/任务/scope拒绝混用。将下方`--state`指向新索引即可。

本地一次性协调器记录的`series-state.json`也是执行索引，不属于公共数据。单次run的`manifest.json`、`DONE.json`、`integrity.json`、逐条预测和评分summary才是结果证据。汇总工具从索引检查完整ID、预测SHA和已完成状态：

```bash
"$EVAL_PY" scripts/bench_v2/verify_runs.py --state /path/to/series-state.json --output /path/to/new-verification
"$EVAL_PY" scripts/bench_v2/aggregate.py --state /path/to/series-state.json --output /path/to/new-analysis --repeats 1000
"$EVAL_PY" scripts/bench_v2/diagnose.py --state /path/to/series-state.json --output /path/to/new-diagnostics
"$EVAL_PY" scripts/bench_v2/report_tables.py --state /path/to/series-state.json --results /path/to/new-analysis/results.json --output /path/to/new-tables
```

公共发布保留科学结果、ID、数据源、配置和SHA，不上传模型、原始文章、完整用户提示或服务器队列/PID。原始逐条生成在研究服务器留档；公开文件可独立重建相同任务并重新生成，但本仓库不分发训练模型权重。
