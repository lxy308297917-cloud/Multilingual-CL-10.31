# 2026-11-08研究结果交付

请先读[给ChatGPT的完整交接](chatgpt_handoff.md)。

- [三语言结果总览](aya-4b-generalization-20261107/overview.md)
- [AR/RU完整报告](aya-4b-generalization-20261107/report.md)
- [0/3-shot翻译开发诊断](translation-controls-20261106/a-report.md)
- [Raw/Clean/混合训练对照](translation-three-lines-20261106/report.md)
- [NLLB替换实验](translation-nllb-20261106/README.md)
- [原4B全部方法与指标](sw_original_all_scores.csv)
- [模型、预算与核验汇总](scientific_record.json)

所有新组固定Final、单seed42；负结果、未运行项、正式测试暴露及测量限制见交接。未新增GPU实验。

## 配置与代码用途

各结果目录中的configs保留运行时科学参数。`LOCAL_RESOURCE`是删除服务器私有路径后的**不可直接执行**占位符；使用前要分别映射模型、数据、tokenized cache及独立输出目录，不能把所有占位符替成同一路径。数据及权重未打包。XL-Sum配置中run_status等准备期状态不代表最终运行状态，已移除；最终结果以报告和数值表为准。

LoRA训练复用仓库已有的`scripts/aya_4b_20261105/train_revised.py`及LLaMA-Factory，不另维护训练框架。新增`scripts/aya_4b_generalization/score_summary_ar_ru.py`与`scripts/unified_language_cl/summary_scoring_v2.py`公开原生文字ROUGE计分逻辑，仅将依赖入口改为仓库相对路径/环境变量，不修改计分公式。

评分依赖为XL-Sum的ROUGE分支，固定上游commit `afcd803c8d7c98f3be60e9a7ce57fcbc3e729e8c`，以及对应分词依赖、NLTK资源、SacreBLEU。须配置`XLSUM_ROUGE_ROOT`（包含rouge_score包的目录）、`METRICS_ADDONS_ROOT`和`NLTK_DATA`。对照验证命令：`python scripts/aya_4b_generalization/score_summary_ar_ru.py --validate-only`。公开导出版本做语法及原环境评分oracle验证；未重新训练或重复正式测评。

主机中断原日志和逐题输出保留本地，不将完整服务器运行记录打包。公开manifest记录本次文件hash和原始来源hash；路径脱敏、移除准备期状态/样本ID清单等变更不改变数值表。原有GitHub文件和历史结果保持原样。
