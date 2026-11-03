# 精简公开复现说明

本轮自动发布审查拒绝包含服务器资产清单、运行计划及环境清单的较完整版本。这个公开版移除了上述内容；完整执行原稿、原始日志、数据/模型及内部清单保留在本地。科学参数、全部汇总结果、区间与曲线不变。路径参数化的公开源码与执行原稿分别记录哈希，不冒称开箱即用的跨机器镜像。

## 训练

官方后训练Qwen/Qwen3-1.7B revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e。配置为`configs/next_stage/ha_only_6400_fft_seed42_20261103_v1.yaml`；仅资产/产物位置改为项目相对角色路径。其他参数与执行原稿相同。必要数据SHA、实际token/样本数、所有正式任务backend/fewshot/batch/model参数见`configs/next_stage/scientific_protocol.json`。既有栈锁为`configs/sft_v1/stack.lock.json`。

原生入口：设置两张经检查空闲的卡，再执行`python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 scripts/sft_v1/train_formal.py --config configs/next_stage/ha_only_6400_fft_seed42_20261103_v1.yaml`。须准备相同已锁定资产及数据缓存，并为新运行指定独立输出；本公开版没有启动器、队列、服务或运行计划。

## 开发与正式评价

`evaluate_dev.py`接受`--model --data --output --label --step`，使用同一冻结编码材料；`metrics.py`实现完整因果续写、四选项概率/margin、移位assistant-token NLL。编码问题/答案和原始逐题记录不分发，协议哈希用于指认本地材料。知识114/0shot与正式14042/5shot分开；阅读180已曝光，不用于选Final。数据冻结早于新开发输出、晚于固定配方主训练启动，如实记录。

正式固定Final使用原生HF Harness的HA/EN/SW阅读、知识任务和原vLLM核心任务，全部科学参数见scientific_protocol.json，不发布包含服务器内部位置的命令清单。既有数据准备/任务代码在`scripts/research_my_method/`和`scripts/sft_v1/`。只评固定Final，不据test改配方。

`analyze_curves.py`、`analyze_ha_control.py`、`audit_formal_option_bias.py`、`audit_reading_runtime_sensitivity.py`读取本地保留的完整产物生成公开汇总；不修改主分数。`audit_ha_data.py`公开版以`--dataset`显式传入数据根目录。`plot_curves.py`生成PNG/PDF。代码需要自行准备锁定的本地输入，未宣称没有逐题原稿也可重算所有分析。

## 验真

CPU检查为`python -m unittest discover -s tests/next_stage -v`，共5项。另有真实Base342预测校准、540阅读输入/2160续写编码一致、全部原生ACC重算、6400曝光完全对齐及真实4步对2+2步恢复的聚合回执。`resume_probe_v2.py`是实际工程探针的可配置入口，使用原生训练；主400步没有恢复。

完整资产/环境清单、服务器运行控制器及源文件绝对位置不在此公开版本。结果JSON中的本地输入清单以单一SHA指认，不公开清单。数据语义质量仍未由独立双语人工认证；位置偏好是观测，不是已定位的因果机制。
