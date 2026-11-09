# XQuAD补测入口

- `prepare.py`与`utils.py`：固定官方AR/RU各1190题、240段文章，标准Harness原生语言提示；官方SQuAD v1.1全部参考最大EM/F1。SW不支持，不合成替代。
- `schedule.py`：只读取已存在或已在运行的Final400模型。四个尚未训练、由用户暂缓的模型不启动；共10个模型语言单元。
- `evaluate.py`：原LM Evaluation Harness `generate_until`，0-shot、non-thinking、greedy128、换行停止、batch4；复用统一账号上限5的安全交接器，不占空卡。
- `report.py`：原始EM/F1逐题复算、240段文章分簇配对区间及输出格式诊断；后验格式统计不改变官方评分、提示或抽取答案。
- `observe.py`：有限期CPU刷新报告，不启动训练。

结果在research/xquad_20261109，数据/模型/评分环境哈希在configs/xquad_20261109。XQuAD保持补充身份，不并入固定5+5平均。不要把原始EM涨幅全部解释为语义理解增强；同时检查答案前缀、长度和F1。

全部10端点完成后，可运行 `python scripts/xquad_20261109/verify_numeric.py` 独立复算公开数值向量的20项分数和42项配对区间；不需要GPU或原始题文，不替代原始官方评分审计。
