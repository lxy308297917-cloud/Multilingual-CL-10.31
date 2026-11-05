# MathOctopus 每语言独立合格记录

来源 revision `dd8076b83bacd7c6ade2829317e8a500c6ffbec4`；原始文件 SHA256 `57864b498f8ea0bc0f8903e8fd912c079ea1a824fad23b1077a0149b455f5022`。English-led block 重建源题身份，7473 个锚点精确匹配官方 GSM8K train。每语言独立过滤，不要求九语言交集，不设任意条数上限，不扣除旧内部256题dev。

|语言|原始|合格独立源题|排除|
|---|---:|---:|---|
|sw|7472|6915|{'final_numeric_disagrees_with_original_gsm8k_gold': 556, 'unparseable_final_numeric': 1}|
|bn|6539|6058|{'final_numeric_disagrees_with_original_gsm8k_gold': 445, 'unparseable_final_numeric': 36}|
|th|7473|6908|{'final_numeric_disagrees_with_original_gsm8k_gold': 557, 'unparseable_final_numeric': 8}|
|ja|7471|7024|{'final_numeric_disagrees_with_original_gsm8k_gold': 447}|
|zh|7364|6898|{'final_numeric_disagrees_with_original_gsm8k_gold': 465, 'unparseable_final_numeric': 1}|
|ru|7361|6766|{'final_numeric_disagrees_with_original_gsm8k_gold': 518, 'unparseable_final_numeric': 77}|
|fr|7469|6763|{'final_numeric_disagrees_with_original_gsm8k_gold': 705, 'unparseable_final_numeric': 1}|
|de|7466|6629|{'final_numeric_disagrees_with_original_gsm8k_gold': 639, 'unparseable_final_numeric': 198}|
|es|7470|6926|{'final_numeric_disagrees_with_original_gsm8k_gold': 543, 'unparseable_final_numeric': 1}|

数值一致不证明翻译语义与整条推导正确；语言/推导抽样核验待做。仅排除已知精确题文/源题重叠，未声称绝对无污染。对齐身份是重建而非上游显式题ID。各语言源文件与合格记录哈希见机器审计。

## 有界语言核验及保留的混合语言上下文

每语言按固定SHA顺序抽32条，GlotLID仅检测语言存在性；不足50字母的文本不计检测。它不能证明翻译语义或推导正确，也未据此过滤冻结记录。

|语言|题文匹配/可测|回答匹配/可测|题文与原始英语源题完全相同的记录|
|---|---:|---:|---:|
|sw|31/32|30/30|13|
|bn|32/32|24/24|60|
|th|32/32|28/28|30|
|ja|31/31|21/21|0|
|zh|17/17|16/16|28|
|ru|32/32|30/30|0|
|fr|28/32|27/29|418|
|de|32/32|28/29|66|
|es|32/32|28/30|64|

原始发布存在少量英语题文，法语较多（418条），不能声称所有题文已正确翻译。训练请求语言与assistant回答才是发布语言标签；本轮保留数值合格但上下文可能为英语的记录并明确其分布，未仅凭自动语言识别删除数据或按benchmark结果改变数据。SW/BN/TH抽样可测回答均识别为目标语言；这不能认证全库。后续结论需考虑翻译噪声及任务分布失配。明确的最终数值错误已按预定规则排除。

## 小样例推导阅读核验

对SW/BN/JA各按固定SHA顺序读取3条，共9条目标语言记录，涉及4个不同英语源题组（跨语言复用原题，不冒充9个独立来源）。对照官方英语参考及目标链，所查百分比、卡路里、丢失物品、包装胶带计算未发现明确算术不一致。

仍观察到词汇/语义翻译风险，例如对象名称偏移、体育参与措辞偏移；算术可保持正确，但不能据此认证原题语义或母语流畅性。记录为有界质量核验，不估计全库错误率，不据此重筛冻结数据。原文只保留私有审计，公开仅源题哈希、样本数量和判读局限。

过滤解释：数值不可解析或与原始参考不一致属于训练前规定的保守可验证条件，不等同于已经逐条证明被排除记录的语义/数学结论错误（例如不同表达形式也可能被保守排除）。追加的 #### 标记只在原回答末数已与原始参考一致时添加，不合成新的推导。

## 原始英语来源的补充精确核对

已额外核对英语GSM8K test1319、MGSM-Rev2 250、MSVAMP1000和同版本PolyMath500题文，与合格训练记录的原始英语源题身份比较，九语言均未命中精确规范化英语原题。只按NFKC/空白/大小写规范化后的完整题文身份判断，不把相同答案当作同题。结果及额外英语来源哈希单独记录，原冻结训练缓存及69文件评测锁不变；没有新增英语PolyMath效果评测，也不宣称无语义/改写污染。
