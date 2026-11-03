# SST 上游与本机适配审计

## 已核实来源

- 论文：arXiv:2610.22293v1（2026-10-15），第2.1–2.3节、附录 Algorithm 1、B.13。
- 官方仓库：https://github.com/Arnav0400/SST
- 实际 commit：`ff0c5bfb32f621ca952d12042b046251181b8aa5`（2026-10-22）。
- 已阅读 `sst/methods/sst.py`、`sst/training.py`、`sst/methods/README.md`、`configs/method/sst.yaml`。

公开实现约束累计参数差，使用固定参考的精确 thin SVD，保护七类 Attention/MLP 投影；其他可训练参数不受 SST 约束。公开入口仅实现 NoRefresh；R10 按论文 §2.3 在完成更新投影后每10次真实优化器更新重新锚定。论文 B.13 提及 Gram 特征分解，而本轮先使用精确 SVD，单独报告成本，不用未验证的近似分解。

## 关键工程差异

1. 上游保护比例使用百分数并将秩裁剪至 [1,min-1]；本实现按论文 floor(r min) 定义，支持0/1作数学测试。r=.99且本模型全部指定形状时两者一致。
2. 上游在取/写 master 异常时会回退至 BF16 参数。本项目禁止该回退：不能访问真正 FP32 master 则报错。
3. 上游默认 ZeRO2，本项目 FFT 锁定 ZeRO3。当前 DeepSpeed safe_set_full_fp32_param 只写 FP32 分片，必须同时更新低精度 ds_tensor 分片以及已聚合副本；将通过真实 DeepSpeed/Qwen smoke 验证后才允许正式训练。
4. 当前 Accelerate DeepSpeedEngineWrapper 在 backward 内执行真实 engine.step；HF on_pre_optimizer_step 此时已晚。适配层直接包装 ZeRO optimizer.step，在其前后捕获和约束实际 FP32 参数变化；原始梯度、Adam矩、clip及scheduler仍由原框架计算。
5. 对方阵仅存完整短轴的自由尾基；对矩形长轴存 protected 基，避免遗漏长轴额外零空间。投影与显式双侧投影对照测试通过，不构建大稠密单位矩阵。
6. NoRefresh 使用累计差投影，相对逐步投影仅允许数值误差；R10在更新之后刷新参考。checkpoint保存当前参考/基、真实步数和刷新记录，引用有哈希的初始基文件，结合原生模型/master/optimizer/scheduler/RNG checkpoint恢复。

## 已锁定实验

Qwen3-1.7B 官方后训练 revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；Aya SW6400固定一轮、400步、有效batch16、2048长度、不packing、assistant-only、BF16、seed/data_seed42、cosine/warmup.05、LR1e-5、AdamW、wd0、clip1，与当前FFT配置逐字段核对。SST两组只改变刷新频率。正式测试仅报告400步Final，不用于调参或选模。

## 当前验收

CPU数学/Adam/恢复/无约束测试8项通过，含多形状和多恢复节点；真实双卡DeepSpeed主权重/低精度同步/Adam矩不变测试通过；Qwen3 NoRefresh确定性连续/恢复全部1,720,574,976个FP32值逐位相同，样本/RNG/scheduler一致。非确定性约3e-6偏差失败保留，未放宽原门槛；正式SST启用full_determinism，旧FFT默认非确定性，这项执行差异披露。NoRefresh正式400步已完成，6400样本暴露与旧FFT逐条一致；真实R10连续20步与从10步恢复至20步均退出0，完整数值与刷新状态对比通过：全部1,720,574,976个FP32值、196组参考/左右基逐值一致，样本/RNG/scheduler一致。正式benchmark进行中，core四项与Belebele已完成；其余不能冒充完整结果。

## 实现身份

本轮不是直接执行官方仓库原trainer：以锁定官方NoRefresh算法为依据，在现有LLaMA-Factory/DeepSpeed训练路径中独立适配投影与恢复，保持主项目一套正式训练框架。R10来自论文规则，官方锁定版本尚无R10训练入口。公开交付须同时给出上游commit、适配代码、数学/真实模型验收与差异，不能称为官方原样运行。

## 刷新调度的独立性能检查

原实现每次在两卡之间轮流处理196个SVD，工程实测约391秒。新增独立调度子类将同一世界大小的一组矩阵各交给原owner并行分解，保持gesvd、参数顺序、rank、FP32值、投影/Adam与刷新时刻不变。当前先安排同196个冻结矩阵、真实FP32 NCCL gather的顺序/并行对照，只有所有参考与左右基逐值相同且实测加速>=1.1才用于尚未启动的唯一正式R10，否则保持原实现。该检查尚未完成，不能提前称已加速。原NoRefresh训练入口与正在运行的源文件未修改。

### 调度检查实际结果（02:57）

顺序重算196矩阵耗时391.937秒，但第一矩阵的保存基逐值比较未通过；并行部分未运行。该优化没有启用，正式R10沿用原先已通过Qwen原生恢复的实现。CPU环境审计发现独立入口设置CUBLAS `:4096:8`、未设置CUDA_LAUNCH_BLOCKING，而实际HF `enable_full_determinism`会改为`:16:8`及blocking=1。这是已识别的运行条件不匹配，尚未用GPU证明它是唯一原因，不能称并行算法错误或已得到加速。失败产物保留，不继续占用正式benchmark资源重试。

## NoRefresh正式Final400完成（03:35）

6400样本、2884423输入/1078347监督token，400 optimizer步，wall9956.924825秒。原FFT逐rank完整暴露记录包含顺序、输入/标签哈希、真实更新步及microstep，均逐条一致。实际Final400 master/导出审计通过，196矩阵BF16自由空间外能量比例18.0119%，native master为3.41183e−10。这是数值几何边界，不是下游遗忘证明。正式R10沿原已验证实现运行，未采用失败的并行检查。
