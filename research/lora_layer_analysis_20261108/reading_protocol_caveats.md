# 阅读选择题的选项偏好：解释限制

已审查48/48个预期结果单元；只复用已有候选似然，不重新推理、不修改正式协议。

选项A预测比例高不代表能将其影响从准确率中直接扣除；这是结构性混杂证据，不是新评分。所有准确率仍按原四候选似然argmax计算。

|语言|任务|模型/消融|ACC%|预测A%|真实A%|
|---|---|---|---:|---:|---:|
|sw|Belebele target|Base|27.33|91.67|22.89|
|sw|Belebele target|Full|38.11|56.44|22.89|
|sw|Belebele target|remove_early|42.11|49.56|22.89|
|sw|Belebele target|remove_middle|37.00|68.33|22.89|
|sw|Belebele target|remove_late|38.22|57.22|22.89|
|sw|Belebele target|keep_late|26.56|92.44|22.89|
|sw|Belebele target|remove_attention|42.00|49.11|22.89|
|sw|Belebele target|remove_mlp|41.22|50.78|22.89|
|sw|Belebele en|Base|62.89|54.33|22.89|
|sw|Belebele en|Full|86.33|28.33|22.89|
|sw|Belebele en|remove_early|86.56|28.33|22.89|
|sw|Belebele en|remove_middle|82.22|35.11|22.89|
|sw|Belebele en|remove_late|86.67|27.78|22.89|
|sw|Belebele en|keep_late|71.56|40.78|22.89|
|sw|Belebele en|remove_attention|89.11|26.56|22.89|
|sw|Belebele en|remove_mlp|87.89|26.78|22.89|
|ar|Belebele target|Base|46.22|47.78|22.89|
|ar|Belebele target|Full|78.44|24.11|22.89|
|ar|Belebele target|remove_early|79.00|24.11|22.89|
|ar|Belebele target|remove_middle|79.56|25.44|22.89|
|ar|Belebele target|remove_late|79.44|27.00|22.89|
|ar|Belebele target|keep_late|50.44|48.67|22.89|
|ar|Belebele target|remove_attention|81.78|26.33|22.89|
|ar|Belebele target|remove_mlp|80.78|25.78|22.89|
|ar|Belebele en|Base|62.89|54.33|22.89|
|ar|Belebele en|Full|87.67|25.00|22.89|
|ar|Belebele en|remove_early|87.22|26.00|22.89|
|ar|Belebele en|remove_middle|85.78|27.56|22.89|
|ar|Belebele en|remove_late|88.22|25.22|22.89|
|ar|Belebele en|keep_late|72.89|41.56|22.89|
|ar|Belebele en|remove_attention|89.00|25.00|22.89|
|ar|Belebele en|remove_mlp|88.44|26.00|22.89|
|ru|Belebele target|Base|55.56|37.78|22.89|
|ru|Belebele target|Full|81.89|26.22|22.89|
|ru|Belebele target|remove_early|81.22|29.56|22.89|
|ru|Belebele target|remove_middle|84.67|25.44|22.89|
|ru|Belebele target|remove_late|82.78|26.89|22.89|
|ru|Belebele target|keep_late|56.00|48.56|22.89|
|ru|Belebele target|remove_attention|84.00|26.22|22.89|
|ru|Belebele target|remove_mlp|83.33|28.11|22.89|
|ru|Belebele en|Base|62.89|54.33|22.89|
|ru|Belebele en|Full|89.00|25.78|22.89|
|ru|Belebele en|remove_early|88.11|26.56|22.89|
|ru|Belebele en|remove_middle|88.56|25.56|22.89|
|ru|Belebele en|remove_late|88.67|25.44|22.89|
|ru|Belebele en|keep_late|69.44|48.22|22.89|
|ru|Belebele en|remove_attention|89.33|25.56|22.89|
|ru|Belebele en|remove_mlp|88.56|26.22|22.89|

## 对层级机制解释的影响
Base存在显著答案位置偏好，消融/Full之间的阅读变化也可能包含该偏好的变化。不能把目标阅读ACC上升全部解释为新语言知识增长，也不能由低分直接认定某个层组损伤了语义推理。

因此，层级功能分离结论必须同时报告摘要、GSM8K灵活提取和官方IFEval等生成式指标；这些指标也分别有自身测量边界。当前没有通过独立选项重排或校准实验分离答案格式/位置因素，不能把本次观察升级为纯语言功能局部化的因果证据。

后续如开展选项重排、均衡示例或位置校准，应先在独立开发材料冻结设计；本轮正式测试不用于选择提示、最佳层数或新方法。上述额外推理控制NOT_RUN。

完整混淆矩阵见`reading_choice_audit.json`。正确选项margin仅为辅助诊断，不替代ACC，也不作为方法去留依据。
