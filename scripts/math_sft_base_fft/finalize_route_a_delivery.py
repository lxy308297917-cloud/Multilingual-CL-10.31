"""Render handoff from completed fixed roles; no score/epoch selection or GPU work."""
from profile_paths import resolve
import json
from pathlib import Path
R = Path(resolve('ROOT'))
D = R / 'research/math_sft_base_fft'
S = R / 'results/math_sft_base_fft'

def read(p):
    return json.loads(p.read_text())

def main():
    assert (S / 'routeA-analysis-v1/DONE.json').exists()
    assert (S / 'delivery-evidence-v1/DONE.json').exists()
    assert (S / 'output-language-audit-v1/DONE.json').exists()
    roles = read(S / 'routeA-analysis-v1/all_roles.json')
    decision = read(S / 'routeA-analysis-v1/route_decision.json')
    cost = read(S / 'delivery-evidence-v1/cost_accounting.json')
    pairs = read(S / 'routeA-analysis-v1/general_paired_comparisons.json')
    assert len(roles) == 9 and {r['language'] for r in roles} == {'sw', 'bn', 'ja'}
    costtable = '|语言|每轮源样本|E1步数|E2累计步数|两轮实际曝光|输入token|监督token|训练分钟|训练GPU小时|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for x in cost['training']:
        costtable += f"|{x['language']}|{x['source_records_per_epoch']}|{x['epoch1_optimizer_steps']}|{x['epoch2_optimizer_steps']}|{x['actual_exposures']}|{x['input_tokens']}|{x['supervised_tokens']}|{x['trainer_runtime_seconds'] / 60:.2f}|{x['trainer_gpu_hours']:.3f}|\n"
    epoch_cost_table = '\n## 两个保存节点的累计训练强度\n\n|语言|节点|累计步数|累计实际曝光|累计监督token|回调计时GPU小时|\n|---|---|---:|---:|---:|---:|\n'
    for x in cost['training']:
        for e in x['epoch_checkpoint_costs']:
            epoch_cost_table += f"|{x['language']}|E{e['epoch']}|{e['steps']}|{e['cumulative_exposures']}|{e['cumulative_supervised_tokens']}|{e['cumulative_two_gpu_hours_to_checkpoint']:.3f}|\n"
    epoch_cost_table += '\n节点计时使用该checkpoint两rank记账中的较大进程累计时间乘以2张分配GPU；与训练器train_runtime、完整作业分配墙钟是不同的观测口径，分别保存，不混算为kernel计算时间。\n'
    (D / 'cost_report.md').write_text('# 实际训练强度与成本\n\n' + costtable + epoch_cost_table + f"\n三语言训练器运行合计：**{cost['training_total_trainer_gpu_hours']:.3f} GPU小时**。每项为两张RTX3090分配数乘训练器实际运行时间，包含训练器内保存checkpoint；不是GPU kernel活跃时间，也不包括Base普查、工程验证和正式测评。\n\n" + f"路线A训练/测评队列分配墙钟合计 **{cost['route_A_total_allocation_gpu_hours']:.3f} GPU小时**，包含模型加载、独立验证和阶段CPU等待，与上面的训练器成本口径不同。Base九语言生成、Base通用、工程验证另列机器记录，不能将它们隐去后声称是全部研究成本。CPU等价评分和代码沙箱时间不冒充GPU计算。\n\n" + '所有语言使用全部合格MathOctopus记录、连续两轮、有效batch16。分布式padding重复和尾批均计入实际曝光，不把相同epoch当作相同token预算。实际训练进度日志、数据与权重哈希、每轮完整覆盖验证见delivery-evidence-v1。原训练器total_flos不是可靠的设备实测FLOPs，本报告不据此声称硬件效率。\n\n完整E1/E2权重保存在实际checkpoint存储，未上传GitHub；save_only_model=true，两轮中优化器在同一进程连续运行，但没有落盘优化器状态，不宣称精确断点恢复。工程验证33条×2轮共72次曝光（含DDP padding）单独计量，不计入正式训练收益。路线B状态：' + decision['route_B'] + '。\n')
    table = '|语言|模型|MGSM ACC%|MSVAMP ACC%|PolyMath DW-ACC%|EN GSM8K%|EN MMLU%|IFEval strict%|HumanEval+%|\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for x in roles:
        table += '|' + x['language'] + '|' + x['role'] + '|' + '|'.join((f"{x[k] * (1 if k == 'polymath_DW_ACC_percent' else 100):.2f}" for k in ['mgsm_rev2_acc', 'msvamp_acc', 'polymath_DW_ACC_percent', 'GSM8K_flexible', 'MMLU', 'IFEval_prompt_strict', 'HumanEval_plus'])) + '|\n'
    declines = [x for x in pairs if x['before_epoch'] == 0 and x['paired_bootstrap95_pp'][1] < 0]
    losses = '|语言|节点|通用任务|相对Base变化（百分点）|逐题配对95%区间|\n|---|---|---|---:|---|\n'
    for x in declines:
        losses += f"|{x['language']}|E{x['after_epoch']}|{x['task']}|{x['delta_pp']:+.2f}|[{x['paired_bootstrap95_pp'][0]:+.2f}, {x['paired_bootstrap95_pp'][1]:+.2f}]|\n"
    if not declines:
        losses = '当前四项通用主指标没有出现配对95%区间整体低于零的节点；仍保留全部细小负数，不把缺少明确下降解释为证明没有遗忘。\n'
    status = '路线A达到预先冻结的数学学习门槛，路线B触发条件不成立，未运行。' if decision['status'] == 'PASS' else '路线A在两个固定epoch均未达到预先冻结门槛；已授权的条件路线B尚待独立数据/资源审计和执行，整个研究目标仍未完成。'
    epoch_details = '\n## 第一轮到第二轮具体发生了什么\n\n'
    for lang in ['sw', 'bn', 'ja']:
        rs = {r['epoch']: r for r in roles if r['language'] == lang}
        epoch_details += '### ' + lang.upper() + '\n\n'
        epoch_details += f"E1相对Base：MGSM {100 * (rs[1]['mgsm_rev2_acc'] - rs[0]['mgsm_rev2_acc']):+.2f}个百分点，MSVAMP {100 * (rs[1]['msvamp_acc'] - rs[0]['msvamp_acc']):+.2f}个百分点。\n\n"
        epoch_details += f"E2相对E1：MGSM {100 * (rs[2]['mgsm_rev2_acc'] - rs[1]['mgsm_rev2_acc']):+.2f}个百分点，MSVAMP {100 * (rs[2]['msvamp_acc'] - rs[1]['msvamp_acc']):+.2f}个百分点，PolyMath DW-ACC {rs[2]['polymath_DW_ACC_percent'] - rs[1]['polymath_DW_ACC_percent']:+.2f}个百分点。\n\n"
        epoch_details += 'E2相对E1的通用指标点估计：' + '；'.join((f'{label} {100 * (rs[2][key] - rs[1][key]):+.2f}个百分点' for key, label in [('GSM8K_flexible', 'GSM8K'), ('MMLU', 'MMLU'), ('IFEval_prompt_strict', 'IFEval strict'), ('HumanEval_plus', 'HumanEval+')])) + '。是否有明确变化以[E2−E1逐题配对区间](epoch2_minus_epoch1.md)为准，不能把每个小幅正负差值都解释为可靠改善或遗忘。\n\n'
    epoch_details += '各任务分别判断第二轮的收益与代价，没有定义跨任务综合分数。当前只能描述两个固定预算节点；不能从单条训练轨迹确立最优epoch，也不因某项下降删除第二轮。GSM8K strict的改善可能来自输出格式变化，数学保持以flexible正确率为主；实际解析率与MMLU读出异常核验单独保留。\n'
    text = '# 三语言数学FFT研究结论\n\n' + status + '同一epoch通过语言数：' + str(decision['passing_languages_per_same_epoch']) + '。\n\n' + table + epoch_details + '\n## 目标语言学习与通用能力变化\n\nMGSM/MSVAMP使用最终数值ACC；PolyMath为官方表达式等价与官方难度权重；英语GSM8K使用flexible数值ACC，IFEval为官方prompt strict，代码为实际隔离执行HumanEval+ pass@1。不能把格式strict分数当作主要数学能力分数。完整正确数、其他strict/loose指标及每难度成绩见机器表。\n\n下表仅摘出通用指标配对95%区间整体低于零的节点，其他全部正负变化保留在general_paired_comparisons，不把微小下降自动解释为稳健遗忘：\n\n' + losses + '\n## 判断范围\n\n三个语言各自从同一官方Qwen3-1.7B初始化，模型称Base指未改动的官方已后训练模型，不是另一个Qwen3-Base预训练权重。此处没有顺序语言训练，因此不能用这些结果直接证明跨语言顺序干扰或旧语言遗忘；这里观察的是单语言数学SFT之后的目标数学收益与英语通用能力变化。没有开发My Method。\n\n九语言选取及路线决定使用了已暴露benchmark，是探索性研究。仅seed42；配对区间描述固定模型的题目抽样不确定性，未校正多重比较，不能替代训练种子稳定性或独立盲测。完整E1/E2均报告，未用英语下降早停，也未逐语言挑选最优epoch来凑门槛。PolyMath高难度存在明显长度上限命中，解析率/截断率与完整输出长度另列，按原冻结预算解释结果，不临时扩大解码预算。\n\n## 可复核交付\n\n- [三语言完整节点及三个预算差值](mathoctopus_epoch_comparison.md)、[完整数学成绩](target_math_results.md)、[通用能力成绩](general_retention_results.md)、[E2−E1配对区间](epoch2_minus_epoch1.md)、[路线决定](route_selection_decision.md)。\n- [机器可读所有节点](../../results/math_sft_base_fft/routeA-analysis-v1/all_roles.csv)、[数学逐题配对变化](../../results/math_sft_base_fft/routeA-analysis-v1/paired_comparisons.csv)、[通用逐题配对变化](../../results/math_sft_base_fft/routeA-analysis-v1/general_paired_comparisons.csv)。\n- [训练收益曲线](../../results/math_sft_base_fft/routeA-analysis-v1/figures/target_gain_vs_training.png)、[通用能力变化曲线](../../results/math_sft_base_fft/routeA-analysis-v1/figures/general_retention_vs_training.png)。曲线仅有Base/E1/E2三个真实节点，不冒充密集学习曲线。\n- [成本](cost_report.md)与[真实成本记录](../../results/math_sft_base_fft/delivery-evidence-v1/cost_accounting.json)。\n- [真实输入元数据与指纹](../../results/math_sft_base_fft/delivery-evidence-v1/data_inputs/manifest.json)、[模型/协议/提示/代码的实际运行身份](../../results/math_sft_base_fft/delivery-evidence-v1/actual_run_manifests/manifest.json)。原始元数据哈希与公开路径适配后的哈希分别记录。\n- [输出语言辅助诊断](output_language_diagnostics.md)：固定配对样本的自动语言识别，仅作辅助，不等同于人工确认的语言正确率。\n- [无答案逐题结果清单](../../results/math_sft_base_fft/delivery-evidence-v1/per_item_flags/manifest.json)：按模型与任务保存正确性、提示身份哈希及目标生成诊断，不上传原始题目、答案或生成程序。完整原始输出本地保留，可追溯。\n\n训练器的实际loss、学习率、grad_norm日志及每GPU的PyTorch分配/预留峰值已经导出；参数更新L2仅对验证记录中注明的三个矩阵/向量采样，不冒称全模型更新范数。这些优化诊断不替代正式benchmark。\n\n公开代码是带明确路径配置的实际运行源码整理版，部署路径适配前后哈希单独记录；九语言合格训练数据已经用公开代码真实重建并与原输入逐文件SHA一致。新主机仍需配置环境、模型与原固定评测资源，并重新通过实际代码执行隔离验证，不能把源码或复制的旧证明当作新主机安全验证。\n'
    (D / 'research_conclusion.md').write_text(text)
    (D / 'chatgpt_handoff.md').write_text('# 给ChatGPT的研究交接\n\n' + status + '\n\n请先阅读[研究结论与全节点总表](research_conclusion.md)，再阅读[冻结协议](protocol.md)、[数据审计](mathoctopus_data_audit.md)与[真实训练强度](cost_report.md)。本次是三个独立Base→单语言数学FFT，不是ES→TH或SW→Hausa旧顺序实验。\n\n' + table + '\n讨论重点：目标语言数学收益是否明确；通用能力下降出现在哪些任务与预算节点；第二轮相对第一轮是否仍有额外收益或代价。避免混用旧V1代码Base（新全角色HumanEval使用固定V0）、格式strict与主数学ACC、不同token规模的epoch预算。\n\n路线状态：' + decision['status'] + '；路线B：' + decision['route_B'] + '。完整配对区间、所有节点和限制见研究结论链接。本次没有训练My Method，也不凭这些单语言结果宣称存在顺序跨语言参数冲突。\n')
    readme = D / 'README.md'
    old_readme = readme.read_text()
    old_intro = '当前执行：先完成九语言 Base 数学普查，再按冻结规则选择三种语言，独立从同一 Base 做完整两轮 FFT。没有启动新的顺序多语言或 My Method。'
    new_intro = '路线A已完成九语言Base普查、SW/BN/JA三组独立完整两轮FFT、六个节点全量正式评测和逐题配对分析。路线A状态：' + decision['status'] + '；路线B：' + decision['route_B'] + '。没有启动新的顺序多语言或My Method。'
    readme.write_text(old_readme.replace(old_intro, new_intro))
    print(json.dumps({'final_reports_written': True, 'route_A': decision['status'], 'route_B': decision['route_B']}))
if __name__ == '__main__':
    main()
