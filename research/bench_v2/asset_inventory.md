# Benchmark v2 资产审计

## 模型与历史资产

Base是官方后训练Qwen3-1.7B，revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。FFT、LoRA r16、SSU-Wanda均复用Aya SW6400、固定400步Final。模型、Adapter及tokenizer共11个实体文件已重新计算SHA；六类任务内每个方法的checkpoint身份一致。四份原baseline表、旧FLORES脚本及样本锁、机制诊断报告的8份历史文件未改。

## 数据

FLORES原版devtest每方向1012句，旧500句索引及8份历史输出的原文、参考和提示逐条核验；另512句单列。XL-Sum官方v2.0.0 SW test987、EN test11535；HumanEvalPlus v0.1.10为164题、MbppPlus v0.2.0为378题。CrossSum两个方向各270对，独立语义质量未验证，仅可行性审计。固定revision、ID及归档SHA见配置和精选结果。训练完整字段精确匹配零命中，不代表广泛无污染。

## 实际工具与隔离

推理为Python3.11、torch2.7.0+cu118、vLLM0.9.2+cu118、Transformers4.53.2、SacreBLEU2.6.0、rouge-score0.1.2和langdetect1.0.9。独立代码静态处理使用EvalPlus0.3.1；安装时发现numpy1与最新pyarrow冲突，固定pyarrow17、pandas2.2.3后实际sanitize/AST成功。依赖版本、安装源码与分句资源SHA见environments配置。

Docker执行权限不足；namespace基础隔离可用，但CPU、内存和进程资源限制没有实际生效，因此未执行生成代码，pass@1保持NOT_RUN。没有提升权限或修改共享环境。

## 验证与执行

28项定向测试通过；四模型真实MT smoke成功。初次vLLM V1 spawn未继承UUID兼容映射，修复仅涉及NVML映射的父子进程安装，失败日志保留并在独立目录恢复成功。另有资源竞争在CUDA加载前安全退出，复核空闲资源后恢复；未中断他人进程。

本轮仅使用经核实空闲的资源，临时GPU数量授权不改变全局默认。已完成16组MT/SUM正式分数和8组Code生成/静态结果；34项执行及连通检查全部结束。实际成本、完整性与限制见报告及精选结果。
