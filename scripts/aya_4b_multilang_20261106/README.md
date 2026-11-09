# 共享执行与摘要评分工具

- `gpu_slot.py` 为本机排队适配层，依赖单独配置的账号级 `gpu_queue` 模块及仓库 `preflight_gpu.py`。接口使用快照、进程归属和队列预留，将本账号真实任务与预留合并限制为四张卡；所有实验显式选择GPU UUID。外部账号级工具与服务配置由运行环境提供。
- `score_summary_ar_ru.py` 使用既有 `scripts/unified_language_cl/summary_scoring_v2.py`，后者固定到多语言XL-Sum ROUGE checkout `afcd803c8d7c98f3be60e9a7ce57fcbc3e729e8c`。AR/RU采用对应语言分词且不做stemming，ROUGE-Lsum按既定句级LCS修补协议计算。
- 生成复用 `scripts/aya_4b_20261105/generate_v2_gpu_guard.py`；运行环境、路径和完整科学设置见当前正式实验目录。

科学训练入口与排队适配层分别保留。公开文件中的路径标记需要映射到本机已有模型、数据、Python环境及队列工具。
