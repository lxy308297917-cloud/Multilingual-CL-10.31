# 表示机制与 benchmark 补测科学代码

本轮不训练模型。使用P2公开基线50640bb的四组固定权重。结果在[研究入口](../../research/representation_benchmarks_20261109/README.md)。

## 不需要权重的离线验证

```bash
python scripts/representation_benchmarks/verify_results.py research/representation_benchmarks_20261109
```

仅用Python标准库，读取公开CSV/gzip数据并复核：80,160条候选似然ACC、656条代码通过结果、代码清理后实际执行内容的SHA、36,864条逐例层漂移及133,200条逐对层余弦汇总。不会执行模型代码。公开候选代码仅作为数据保留，真实执行始终需要受限隔离环境。

```bash
python scripts/representation_benchmarks/metrics.py
```

第二条需要PyTorch，只做CPU小矩阵稠密/Gram协方差范数、平移/缩放、零值及余弦对照。

## 重新提取或生成

- `prepare.py`：核验四权重、冻结XNLI主要/固定轮换和HumanEval+输入。
- `prepare_parallel.py`：固定300条FLORES dev与两种输入/三种池化口径。
- `benchmark.py` / `generate_code.py`：复用已有HF Harness和vLLM生成实现；不自由生成XNLI答案。
- `extract.py` / `metrics.py` / `analyze_representations.py`：冻结权重、相同teacher输入、raw block特征；明确区分论文跨层代数适配和逐层样本协方差扩展。
- `analyze_benchmarks.py` / `build_tables.py`：逐题审计、premise聚类配对区间、九项总表与四模型描述性关联。
- `supervise.py` / `mechanism_supervise.py` / `finish_benchmarks.py` / `execute_pending_code.py`：本次手动启动的有限衔接记录，不设自启或自动重试。

这些是实际服务器执行快照，保留顶部既有项目、数据、环境和checkpoint默认路径，迁移时需要映射到自己的资产；不是声称在没有私有adapter/缓存时也能开箱复现。所有输出采用独立目录并拒绝覆盖。GPU使用账号统一上限4、UUID显式选卡和原有gpu_slot/preflight检查，相关通用工具沿用已有项目实现。

执行器复用已有 `scripts/math_sft_base_fft/execute_code.py` 与隔离执行器；来源/修复/资源限制见研究报告，不在此新增宿主执行替代。环境不满足隔离条件时不能运行候选。

`source_identity.json` 区分公开研究基线和非干净服务器checkout；实际文件/数据/权重SHA为执行身份。大型隐藏特征缓存保留在服务器，未上传模型或完整测试题。公开数值验证不等于用CSV重新构建全部高维协方差；后者须使用原哈希缓存或重新提取。

当前仅四模型、单SW、单训练seed；没有新增Loss、lambda搜索、训练种子、Logit Lens或顺序CL训练。
