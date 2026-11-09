# Function-Aware正式实验进度

2026-11-09T20:14:45.150268+08:00

当前目标：SW/AR/RU五方法同预算Final400正式比较。旧条件式P3目标与误转跨层求解方案均已被本目标替代。

三语言主线新增正式训练完成：2/14；历史SW batch2普通LoRA复用。

| 任务 | 状态 | 阶段 | 最近已保存节点 |
|---|---|---|---|
| sw_full | WAITING_FREE_GPU_ACCOUNT_AND_QUEUE_MAX4 | final | None |
| sw_kd | VERIFIED_FINAL400 | final | 400 |
| sw_random | WAITING_FREE_GPU_ACCOUNT_AND_QUEUE_MAX4 | final | None |
| sw_function | WAITING_FREE_GPU_ACCOUNT_AND_QUEUE_MAX4 | smoke | None |
| ar_all | RUNNING | final | 100 |
| ar_full | WAITING_FREE_GPU_ACCOUNT_AND_QUEUE_MAX4 | final | None |
| ar_kd | WAITING_FREE_GPU_ACCOUNT_AND_QUEUE_MAX4 | smoke | None |
| ar_random | WAITING_FREE_GPU_ACCOUNT_AND_QUEUE_MAX4 | smoke | None |
| ar_function | WAITING_FREE_GPU_ACCOUNT_AND_QUEUE_MAX4 | smoke | None |
| ru_all | RUNNING | final | 200 |
| ru_full | VERIFIED_FINAL400 | final | 400 |
| ru_kd | WAITING_FREE_GPU_ACCOUNT_AND_QUEUE_MAX4 | smoke | None |
| ru_random | WAITING_FIXED_CALIBRATION_WEIGHTS | not_started | None |
| ru_function | WAITING_FIXED_CALIBRATION_WEIGHTS | not_started | None |

另有独立FFT补充2组完整训练；它们不计入上述14组，进度见[FFT轨道](../fft_representation_20261109/README.md)。主线和FFT合计16组新增完整训练。

队列等待或进程存活不代表GPU正在计算；实际GPU归属另由助手核验。未获得的分数不填为0。

共同72个特征点已完成，和旧36层末端重合的部分逐值误差为0。CPU检查确认六组映射、均匀损失及梯度正确。

AR/RU官方Global-MMLU材料已准备：各14042题、57学科、5-shot，与SW同一官方revision。

本轮不新增独立短训筛选；两步smoke只查代码和曝光计数，不读取benchmark效果。正式测评不会反馈修改规则。
