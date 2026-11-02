# Benchmark v2 固定协议

2026-11-02，本轮仅扩展四组固定Final评测，不训练、不选checkpoint、不改旧分数。用户授权经核实空闲GPU数量不限，不中断他人，默认队列全局4卡设置不改；仅本次启动显式max-user-gpus8（本机物理8卡）。

## 数据与完整性

- FLORES-200 official tar SHA b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6，完整devtest1012句/方向。ID为官方文件0起始行号，原500索引可精确恢复；四模型8份历史输出的source/reference/prompt逐条相同。完整v2包含旧500，不是独立新测试；另外512单列。同一完整生成中的legacy500-members分数不是旧HF原分数。
- 原3-shot dev索引409/521/656，英文翻译指令English/Swahili名称、历史提示字节保留；3个dev示例与完整devtest无精确文本重叠。旧config的pending标签为历史声明，实际提示通过8份原输出核实。
- XL-Sum v2.0.0 HF revision30fece425f9a3866e04321773ca7a80056d55ca6：SWtest987，ENtest11535，0-shot，SW/EN本地语言指令。两种语言是不同文章，不跨语言直接减绝对ROUGE。
- CrossSum HF revisionb3fb19e9c135397c65f11fc9cf4ce58bf86aa29b：EN→SW270、SW→EN270，检索对齐。与XL-Sumtest共享源URL为6/37、共享目标URL为37/6，原文/参考精确重合数量相同；尚未独立核验跨语言语义质量，因此P1先报告可行性，不作为独立正式证据。
- HumanEvalPlus v0.1.10：164；MbppPlus v0.2.0：378，官方完整EvalPlus筛选集。MBPP base与plus分数均在这378题上，不冒充原MBPP所有划分。
- 测试不供训练、蒸馏、选样或方法开发。只读精确字段与既有6400训练样本比对，不能据无精确命中宣称广泛无污染。

## 推理

Base固定Qwen/Qwen3-1.7B revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e；FFT/LoRA/SSU为既有400步Final。统一Base tokenizer和chat template，enable_thinking=False；LoRA由vLLM原生LoRARequest读取r16 Adapter，base与adapter文件SHA均记录，不静默退回Base。

vLLM0.9.2+cu118 V1、BF16、TP=1、不量化；max_model_len8192，max_input_tokens7168，max_num_seqs32，max_num_batched_tokens8192，GPU memory utilization0.80，seed42。temperature0，top_p1，top_k-1，候选n1。显式generation_config=vllm，停止token151645/151643，无额外文本stop。MTmax_new_tokens512、SUM256、Code1024。旧FLORES为HF直接推理，新v2使用统一vLLM，后端/批量变化披露，不声称与旧500逐位同结果。

摘要只截断文章尾部，保留完整本地语言指令与chat wrapper，所有样本仍评分；记录原文章token和截断标记，不将截断输入称为完整文章输入。MT/Code超长直接报错，不静默丢样本。逐条保存实际prompt、token hash、finish_reason和原始响应。

代码使用EvalPlus0.3.1默认user instruction prefix，加官方problem prompt；不用其未关闭thinking的response-prefill，改用同一Qwen3非思考模板生成完整脚本，兼容差异明确记录。所有模型同一提取：EvalPlus0.3.1 sanitize，再AST静态检查；不重新提示失败模型。

## 评分

SacreBLEU2.6.0：chrF char_order6/word_order2/beta2/lowercaseFalse/whitespaceFalse/eps_smoothingFalse；BLEU13a tokenizer、mixed case、exp smoothing、effective_orderFalse。均为语料级，保存完整signature。保留legacy500-members及supplementary512子集定义。

rouge-score0.1.2：ROUGE-1/2/Lsum，use_stemmer=False，默认小写ASCII字母数字词切分；NLTK English Punkt用于两种语言的句分割后以换行传给ROUGE-Lsum。该句分割不是经验证的SW专属模型，词形和标点边界限制披露，不能跨语言比绝对值。主分数按文章F1宏平均×100，辅助chrF++为语料级。

辅助行为：空回答、原文精确复制、1–5词片段连续重复至少10次、token长度、finish_reason length截断、输入截断。langdetect1.0.9 seed42，至少20字母且4词、置信度≥0.8才给语言诊断；未知保留分母与数量。检测器不是语言真值，不能将语言匹配直接当翻译方向语义正确率。

## Code执行安全（当前NOT_RUN）

Docker daemon权限被拒绝；旧CLI对permission denied可能返回0及JSONnull，须同时检查JSON类型与stderr。bubblewrap可创建namespace，但用户scope的TasksMax/MemoryMax/CPUQuota未实际作用：内核pids.max=max，memory.limit约9e18，cpu quota=-1。不为评测提升权限或安装特权服务，不在宿主执行模型代码。因此本轮先生成全部候选与静态诊断，HumanEval/+/MBPP/+ pass@1全部null、NOT_RUN，runtime/timeout亦未测，不能填0或用AST通过率替代。

## Pilot、恢复与统计

Smoke每任务4条，按固定ID hash seed选择；摘要Pilot每语种128条，所有模型同ID。在查看分数前已冻结。结果PILOT_ONLY与Full分开。Full按全部官方ID排序原序生成；缺失、重复、不符split或方向直接失败。完整样本才VERIFIED_COMPLETE，不以部分输出冒充全量。

恢复要求同一模型/数据/核心代码/提示/解码manifest完全一致，跳过已完成ID；冲突和损坏JSONL不静默忽略。失败产物保留，修复后独立目录，恢复记录清楚。

相同ID配对分析，微调减Base。MT与辅助chrF++按配对样本重采样重新聚合语料级指标，不把句级chrF均值当语料级；ROUGE按同文章配对宏平均。固定bootstrap seed42、1000次、95%percentile区间；仅逐任务/同语种比较，不造加权总分。

## 官方来源

- https://github.com/facebookresearch/flores
- https://huggingface.co/datasets/csebuetnlp/xlsum
- https://huggingface.co/datasets/csebuetnlp/CrossSum
- https://github.com/evalplus/evalplus
- https://github.com/mjpost/sacrebleu
- https://github.com/google-research/google-research/tree/master/rouge
