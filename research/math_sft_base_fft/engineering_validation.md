# 完整两轮工程验证

33条真实SW记录，官方完整1.7B、双卡ZeRO3、microbatch2×world2×accum4。原训练完成6个优化步；E1=3步、E2=6步，两轮各实际36条曝光，包含3条DDP padding重复，所有33条源题均覆盖。末尾优化步为实际小批次，未丢弃源题。两轮使用同一optimizer/scheduler对象，LR未在epoch边界重置。

初版验收错误地要求checkpoint显式保存lm_head.weight与embedding两份，合法的tied alias省略使验证失败。失败日志/退出码保留，未重复训练。修复后逐epoch核实张量形状、文件哈希、抽样参数的真实更新、独立HF CPU加载及前向；lm_head与embedding确实共享内存，E1/E2两份权重均独立加载通过。

这只是工程验证，不计入正式效果，不据此选择语言或学习率。save_only_model=true，节点有完整可推理权重但无可恢复optimizer状态；连续轨迹在同一训练进程内验证。

PolyMath校准复用官方算法：恢复上游遗漏的symeval导出符号，不改等价算法。旧代码沙箱的4GiB默认线程栈使官方multiprocessing.Queue feeder无法创建；专用数学sandbox调整为8MiB栈、4GiB内存硬限和4个线程/进程任务，其余隔离保留，可信队列probe及分数/符号/错答/首boxed/缺boxed/sqrt六项校准均通过。HumanEval+继续使用原独立代码sandbox，未改其限制。
