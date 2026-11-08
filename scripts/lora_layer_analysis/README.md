# LoRA层级消融：科学代码

`make_variants.py`提供实验中的独立副本构建与范数计算。它已在SW/AR/RU真实Final400上完成导出验证：18份生成权重与实际测评使用的权重逐字节一致，756行范数数值一致；验证记录在研究结果目录。

```bash
CUDA_VISIBLE_DEVICES='' python scripts/lora_layer_analysis/make_variants.py \
  --base /path/to/Qwen3-4B \
  --adapter sw=/path/to/sw_final400 \
  --adapter ar=/path/to/ar_final400 \
  --adapter ru=/path/to/ru_final400 \
  --output /path/to/new_ablation_directory \
  --norms
```

- 使用`Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`。
- 输出目录必须不存在；原adapter只读。每个禁用模块仅将B置零，验证其余张量等值。
- 范数计算使用`(alpha/r) BA`的Gram恒等式，没有把A或B自身范数当作有效更新。
- 本实现支持本研究的标准LoRA，不支持DoRA/RS-LoRA、rank/alpha pattern、可训练bias或额外保存模块。
- “保留Adapter参数量”指推理中仍有非零增量的范围，不是新训练的参数预算。零化B不减少主干推理FLOPs。

`norm_report.py`与`plot_results.py`读取仓库内`research/lora_layer_analysis_20261108/`的科学结果，生成范数和逐项性能变化图。它们是实际分析脚本的路径适配副本，计分公式不变。

正式测评复用已有lm-eval-harness、PEFT和vLLM代码；运行配置/版本/任务形式见研究报告。不要把这个CPU重建工具视为新的训练框架或LinguaMap Last-k训练复现。权重、原始文本数据及服务器调度器未打包。

## 从本地保留的逐题产物复算

`collect.py`是本次统计代码的路径适配副本；需要未打包的完整逐题产物，而不是只读取汇总ACC。设置三处根目录后运行：

```bash
export ABLATION_LOG_ROOT=/path/to/this_study_outputs
export REFERENCE_SW_LOG_ROOT=/path/to/original_sw_outputs
export REFERENCE_GENERALIZATION_LOG_ROOT=/path/to/ar_ru_outputs
python scripts/lora_layer_analysis/collect.py
```

`ABLATION_LOG_ROOT`下按`formal/<sw|ar|ru>/<mask>/`存放`hf/results`、`core/results`、`summary-scores`及SW的`gmmlu/results`。脚本沿用原产物的文件名/相对目录约定，逐题核对`doc_hash/prompt_hash/target_hash`，摘要核对相同输入/参考及提示指纹。缺完整组会明确标记PENDING，不能算完成结果。历史SW和AR/RU根目录中的参考位置由代码列出；这里提供的是可审查的统计接口，不是包含原始数据和权重的一键重跑包。

`exploratory_interactions.py`在18组主结果完整后，复算已有四个掩码的条件交互，不新增推理。它是事后探索，不能用于挑选训练层数；其声明时间与范围见研究目录中的探索计划。IFEval指令级区间按提示cluster抽样，其余按题目/文章配对；2000次bootstrap，单seed42，未做多重校正。

## 实际测评命令与公开代码位置

研究目录中的`evaluation_commands.json`从60个正式推理阶段的真实argv提取，仅将模型、环境、数据、项目、输出目录和GPU标识替换为占位符。18条CPU摘要评分命令由冻结入口重建，并单独标注来源。文件不包含服务器进程、其他用户信息或调度记录。命令以参数数组保存，不能直接当作shell字符串执行。

命令保留了原工程的相对路径；公开仓库对部分代码采用精选目录。复查或重建时按下表对应，读取YAML中的资源占位符并准备相同冻结数据：

|实际命令中的工程路径|公开文件或目录|
|---|---|
|`configs/aya_4b_generalization_20261107/tasks`|`research/updates_20261108/aya-4b-generalization-20261107/configs/tasks`中的Belebele YAML|
|`scripts/aya_4b_multilang_20261106/score_summary_ar_ru.py`|`scripts/aya_4b_generalization/score_summary_ar_ru.py`|
|`configs/sft_v1/eval_tasks`中的GSM8K、IFEval|`configs/aya_4b_20261105/tasks/core`中的对应冻结配置|
|`scripts/aya_4b_20261105/generate_v2_gpu_guard.py`|同名公开路径；实际生成逻辑在同目录`generate_v2.py`|
|`scripts/bench_v2/score.py`、`scripts/sft_v1/harness.py`|同名公开路径|

SW Global-MMLU的57学科配置见`configs/aya_4b_20261105/tasks/global_mmlu_full_sw`；其原include目录用`LOCKED_EVAL_TASKS`表示。数据包、模型权重、环境和GPU选择均由复查者在本机准备。这里提供可审查的科学配置与接口，完整运行仍需要这些外部资产；不要把argv占位符或原私有目录当成已经打包的资源。
