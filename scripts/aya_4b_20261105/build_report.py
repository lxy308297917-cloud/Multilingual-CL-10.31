"""Render scientific results only after all fixed-budget tasks and audits finish."""
from profile_paths import resolve
import argparse, json, csv, datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = Path(resolve('ROOT'))
ROLES = ['base', 'fft', 'lora', 'ssu']
LABEL = {'base': 'Base', 'fft': 'FFT', 'lora': 'LoRA', 'ssu': 'SSU-Wanda'}
COLORS = {'base': '#666666', 'fft': '#0072B2', 'lora': '#009E73', 'ssu': '#D55E00'}
MAIN = [('Belebele SW', 'ACC', 'SW阅读 ACC'), ('Global-MMLU SW', 'ACC', 'SW知识 ACC'), ('Summary SW', 'ROUGE-Lsum', 'SW摘要 ROUGE-Lsum'), ('Translation EN->SW', 'chrF++', 'EN→SW chrF++'), ('Translation SW->EN', 'chrF++', 'SW→EN chrF++'), ('Belebele EN', 'ACC', 'EN阅读 ACC'), ('MMLU EN', 'ACC', 'EN知识 ACC'), ('Summary EN', 'ROUGE-Lsum', 'EN摘要 ROUGE-Lsum'), ('GSM8K flexible-extract', 'ACC', 'GSM8K flexible ACC'), ('GSM8K strict-match', 'ACC', 'GSM8K strict ACC'), ('IFEval', 'prompt_level_strict_acc', 'IFEval prompt strict'), ('IFEval', 'inst_level_strict_acc', 'IFEval instruction strict')]

def write(p, s):
    p.write_text(s)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    done = json.loads((a.analysis / 'DONE.json').read_text())
    assert done['status'] == 'COMPLETE' and done['score_rows'] == 80 and (done['paired_comparisons'] == 60)
    assert not done['uncompleted_groups']
    a.report.mkdir(parents=True, exist_ok=False)
    fig = a.analysis / 'figures'
    fig.mkdir(exist_ok=False)
    scores = json.loads((a.analysis / 'scores.json').read_text())
    paired = json.loads((a.analysis / 'paired_comparisons.json').read_text())
    lookup = {(x['model'], x['task'], x['metric']): x for x in scores}
    delta = {(x['model'], x['task'], x['metric']): x for x in paired}
    data = json.loads((R / 'configs/aya_4b_20261105/data.lock.json').read_text())
    training = json.loads((a.analysis / 'training_audit.json').read_text())
    cost = json.loads((a.analysis / 'cost_report.json').read_text())
    absolute = ['|任务／指标|Base|FFT|LoRA|SSU-Wanda|', '|---|---:|---:|---:|---:|']
    for task, metric, label in MAIN:
        absolute.append('|' + label + '|' + '|'.join((f"{lookup[r, task, metric]['score']:.2f}" for r in ROLES)) + '|')
    interval = ['|任务／指标|方法|相对Base变化|配对95%区间|', '|---|---|---:|---|']
    for task, metric, label in MAIN:
        for role in ROLES[1:]:
            x = delta[role, task, metric]
            interval.append(f"|{label}|{LABEL[role]}|{x['delta']:+.2f}|[{x['ci95_low']:+.2f}, {x['ci95_high']:+.2f}]|")
    diagnostic = json.loads((a.analysis / 'generation_diagnostics.json').read_text())
    diaglines = ['|模型|任务|样本|触及生成长度上限|空输出|输入文章截断|', '|---|---|---:|---:|---:|---:|']
    for d in diagnostic:
        diaglines.append(f"|{LABEL[d['model']]}|{d['task_group']}|{d['samples']}|{d['generation_hit_cap_count']}|{d['empty_prediction_count']}|{d['article_truncated_count']}|")
    write(a.report / 'generation_diagnostics.md', '# 生成长度与截断诊断\n\n诊断不参与方法选择或调参；触及长度上限可能同时反映长答案、重复输出或截断，不能单独认定原因。所有模型使用冻结greedy协议，保留原始输出供本地复核。\n\n' + '\n'.join(diaglines) + '\n')
    write(a.report / 'results.md', '# 固定Final正式结果\n\n生成测评统一使用内存配置v2：MT/摘要batch8、最多8序列、4096预填充token、显存比例0.65；GSM8K/IFEval最多8序列、8192批量token、显存比例0.65。所有模型沿用相同题目、提示、解码长度和评分；v1尝试保留，未混入本表。\n\nACC、IFEval为百分比；摘要与翻译为指标分，所有指标越高越好。\n\n' + '\n'.join(absolute) + '\n\n## 相对Base变化\n\nACC/IFEval变化单位为百分点，其余为指标分。区间未经多重比较校正，仅反映当前seed42模型的样本不确定性。Belebele按共享段落聚类，IFEval指令指标按prompt聚类；翻译重采样后重算语料级分数。\n\n' + '\n'.join(interval) + '\n')
    target = [MAIN[i] for i in range(5)]
    general = [MAIN[i] for i in [5, 6, 7, 8, 10]]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    for filename, panels, title in [('target_scores', target, 'SW and directional translation: fixed Final400'), ('general_retention', general, 'General-task change from the same 4B Base')]:
        figure, axes = plt.subplots(2, 3, figsize=(14, 7.7))
        axes = axes.ravel()
        for ax, (task, metric, label) in zip(axes, panels):
            if filename == 'target_scores':
                vals = [lookup[r, task, metric]['score'] for r in ROLES]
                ax.bar([LABEL[r] for r in ROLES], vals, color=[COLORS[r] for r in ROLES])
                ax.set_ylabel('Accuracy (%)' if metric == 'ACC' else metric)
                for i, v in enumerate(vals):
                    ax.text(i, v + 0.015 * max(10, max(vals)), f'{v:.2f}', ha='center', fontsize=9)
                ax.set_ylim(0, max(vals) * 1.18 + 1)
            else:
                vals = [delta[r, task, metric]['delta'] for r in ROLES[1:]]
                lo = [v - delta[r, task, metric]['ci95_low'] for r, v in zip(ROLES[1:], vals)]
                hi = [delta[r, task, metric]['ci95_high'] - v for r, v in zip(ROLES[1:], vals)]
                ax.bar([LABEL[r] for r in ROLES[1:]], vals, color=[COLORS[r] for r in ROLES[1:]], yerr=np.array([lo, hi]), capsize=4)
                ax.axhline(0, color='black', linewidth=0.8)
                ax.set_ylabel('Change (percentage points)' if metric != 'ROUGE-Lsum' else 'Change (ROUGE points)')
            ax.set_title(task + ' / ' + metric)
            ax.tick_params(axis='x', labelrotation=12)
            ax.grid(axis='y', alpha=0.2)
        for ax in axes[len(panels):]:
            ax.set_visible(False)
        figure.suptitle(title, fontsize=15)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        figure.savefig(fig / (filename + '.png'), dpi=180, bbox_inches='tight')
        figure.savefig(fig / (filename + '.svg'), bbox_inches='tight')
        plt.close(figure)
    observations = []
    evidence = {}
    for role in ROLES[1:]:
        gains = []
        declines = []
        uncertain = []
        for task, metric, label in MAIN:
            x = delta[role, task, metric]
            (gains if x['ci95_low'] > 0 else declines if x['ci95_high'] < 0 else uncertain).append({'task': task, 'metric': metric, 'delta': x['delta'], 'ci95_low': x['ci95_low'], 'ci95_high': x['ci95_high']})
        evidence[role] = {'positive_interval': gains, 'negative_interval': declines, 'interval_crosses_zero': uncertain}
        target_changes = '；'.join((label + ' ' + f"{delta[role, task, metric]['delta']:+.2f}" for task, metric, label in MAIN[:5]))
        general_drops = [x for x in declines if x['task'] in ['Belebele EN', 'MMLU EN', 'Summary EN', 'GSM8K flexible-extract', 'IFEval'] and x['metric'] not in ['inst_level_strict_acc']]
        drop_text = '、'.join((x['task'] for x in general_drops)) if general_drops else '当前主要通用指标未见配对区间完全低于零的项'
        observations.append(f'- **{LABEL[role]}**相对同一4B Base：{target_changes}。通用能力配对区间低于零：{drop_text}。指标单位和区间见下表；这不是跨种子结论或选模门槛。')
    (a.analysis / 'research_findings.json').write_text(json.dumps({'status': 'DESCRIPTIVE_FIXED_FINAL_RESULTS', 'methods': evidence, 'method_selection_or_early_stop_based_on_tests': False, 'limitations': ['Single training seed; conditional sample intervals do not establish cross-seed stability', 'Model scale and data changed together relative to1.7B historical runs', 'Learning rates and parameter budgets differ across methods', 'One-stage SW adaptation, not a completed sequential multilingual CL study']}, indent=2) + '\n')
    timing = ['|方法|步数|实际样本曝光|训练器耗时（分钟）|训练进程耗时（分钟）|', '|---|---:|---:|---:|---:|']
    for role in ROLES[1:]:
        tr = training['methods'][role]
        c = cost['training'][role]
        timing.append(f"|{LABEL[role]}|{tr['global_step']}|{tr['samples']}|{c['trainer_runtime_seconds'] / 60:.1f}|{c['process_wall_seconds'] / 60:.1f}|")
    report = '# Qwen3-4B：改进SW指令数据后的固定预算基线\n\n## 研究问题\n\n将原模型换为官方后训练Qwen3-4B，并将此前Aya混合改为阅读、正向摘要、知识/常识和双向翻译，检验SW下游收益与通用能力变化。模型固定revision `1cfa9a7208912126459214e8b04321603b3df60c`；不是Qwen3-4B-Instruct-2507，也不是预训练Base变体。\n\n所有模型均从同一未微调参考初始化，固定Final400，不依英语退化早停，不使用正式测试选择数据、超参数或checkpoint。本轮仅一个seed42。\n\n## 实际结果\n\n' + '\n'.join(absolute) + '\n\n' + '\n'.join(observations) + '\n\n完整区间见[逐任务结果](results.md)，输出上限、空输出和文章截断见[生成诊断](generation_diagnostics.md)；全部80项预定主/辅指标和60项配对比较见结果目录。额外的摘要chrF++等原始评分保留，不以单个平均数掩盖方向差异。\n\n## 数据与训练\n\n6400条：原生上下文阅读1920，正向摘要1920（XL-Sum1280、WikiLingua640），知识/常识1280，双向翻译各640。翻译两方向共享640个原始平行句组；不等于1280个独立句对。输入2,373,891 tokens，移位回答监督324,892 tokens；10条阅读上下文采用确定性答案覆盖窗口，入选完整SFT样本无截断。训练指令无规范化重复。\n\n与旧混合相比，去除了标题扩写、反向背景生成、对话和释义判断等来源。旧混合为2,884,423输入、1,078,347回答监督tokens，因此这是任务构成改进，不是监督token完全匹配的因果对照。各任务token占比不等于loss或梯度贡献。\n\n训练1epoch/400optimizer steps，有效batch16，microbatch1×accumulation16，assistant-only、non-thinking、BF16、长度2048、不packing。FFT/SSU LR1e-5；LoRA rank16/alpha32/dropout0.05、LR1e-4。三方法同一6400条实际输入/标签、同一步内配对和全量顺序均经曝光指纹核验。\n\nSSU基于当前4B与同一英语Tulu原文重新校准500×2048tokens；实际保护1,816,796,928/4,022,468,096，即45.1662%唯一元素，不是全模型50%。保存后的受保护元素变化检查单列证据；该工程正确性不等于下游保护效果。校准继承数学偏置，且方法学习率和可更新参数预算不同，不能把所有方法差异归因于保护机制。\n\n## 评测与评分\n\nBelebele EN/SW各900题；EN MMLU和SW Global-MMLU各14042题/57学科/5-shot；FLORES两个方向各1012句/3-shot；XL-Sum SW987、EN11535；GSM8K1319题/5-shot；IFEval541prompt/834instruction。选择题按四个候选答案似然ACC；生成greedy、non-thinking，MT输出上限512、摘要256、数学/IFEval1280。MT/SUM沿用vLLMV1，原Aya core沿用V0，各模型同协议。\n\n选择题从原始四选项分数与gold独立复算；GSM8K按锁定官方strict/flexible提取与exact-match重执行；翻译重算语料级chrF++/BLEU；IFEval核对原运行的官方逐prompt/逐instruction标记与汇总，并重执行确定性检查。官方IFEval对无效参数含随机fallback，语言检测也可能不确定，未凭脱离原RNG状态的事后重评分覆盖主分数。\n\n95%区间为未经多重校正的探索性配对区间，条件于当前seed42权重；不代表训练种子稳定性。英语不同任务可上升也可下降；双向翻译分别解释。正式benchmark已暴露，仅作本轮冻结比较，不按这些分数挑选下一轮配置。\n\n## 数学分数的提取敏感性\n\n同一批1319条GSM8K输出中，flexible判对而strict判错的数量：Base310、FFT448、LoRA283、SSU522。SSU的strict从61.03%降至42.08%，而flexible从84.46%降至81.58%，差异说明输出格式与答案提取规则影响较大；不能把strict全部降幅解释为数学推理能力损失，也不能据此断言推理完全保持。两套官方分数均保留。逐模型交叉统计见`gsm8k_extraction_diagnosis.json`。\n\n## 成本与产物\n\n' + '\n'.join(timing) + '\n\n本机GPU为24GB级显存。FFT/SSU采用CPUAdam offload，仍需对4B模型执行密集前后向和优化器更新；SSU掩码不等于减少相同比例的计算或优化器存储。回答监督虽为324,892 tokens，完整上下文输入仍为2,373,891 tokens。\n\n每个模型另评29,884道选择题（约119,536个候选似然请求）和16,406条生成任务。四模型累计约478,144个候选似然请求、65,624条生成；题目长度不同，不能将条数当作相同计算量。训练器耗时、进程墙钟和GPU显存采样有不同覆盖范围，均不冒充CUDA kernel时间。采样峰值是观测下界；完整计算口径、失败尝试、模型/数据/配置哈希和逐项评分证据在机器可读审计中保留。空间有限，仅导出完整Final及独立smoke权重，不保存optimizer checkpoint，不能宣称从这些权重精确恢复优化器轨迹。\n\n## 后续翻译与知识问题：已完成的有限诊断\n\nSW知识只有SSU的小幅正向变化；FFT和LoRA没有建立收益，不能因为阅读提高就推断知识提高。原数据HotpotQA/PIQA覆盖与Global-MMLU的57学科任务有差异，但当前未用隔离实验验证该差异的因果贡献。\n\n原EN→SW存在大量重复和触及512输出上限的结果。已核验全部6400条训练的终止符、回答mask与non-thinking前缀，没有发现对应错配；这些检查不证明数据质量或训练配方没有其他问题。解码开发对照能缓解部分退化，却不能单独证明训练学到了翻译。\n\n额外正式冻结权重的XNLI/XCOPA补充结果已经完成。XNLI的零样本指令选择题收益对示例格式敏感：192题/语言的开发对照中，Base加入3个均衡示例后大幅改善；FFT相对Base的SW增益由零样本+10.94个百分点变为3-shot的+5.73个百分点，后者区间跨零。因此不将零样本增益解释为已确认的纯语义能力提升。补充任务在训练结束后冻结，属于探索性扩展，不冒充训练前预注册的主要任务。\n\n随后一个数据重平衡LoRA完成相同400步，但监督token变为209925，不能称为与原混合等token预算。固定64组已暴露FLORES dev、官方non-thinking采样下，新LoRA相对同批Base的EN→SW为+1.13 chrF++（95%区间[-2.79,+4.90]），SW→EN为+0.47（[-1.93,+2.74]）。区间均跨零，未达到预定继续投入条件，尚不能声称修复成功。greedy下EN→SW仍落后；完整两种解码均保留。该候选的正式test及各保持任务NOT_RUN；第二个LR2e-5候选NOT_RUN。不能把未测保持任务描述为没有遗忘。\n\n[数据重平衡候选详细结果](translation_rebalanced_candidate_result.md)保留额外训练成本和开发判定；原四模型表不覆盖。后续应先确定任务匹配的可靠收益，再决定扩大翻译修正或研究保护方法，不以更换benchmark掩盖原翻译负面结果。\n\n## 可以支持和不能支持的结论\n\n当前证据仅支持上表所示固定模型的逐任务变化。模型规模与训练数据同时变化，不能分别归因；也不能把阅读或摘要改善外推为所有SW任务改善。没有开展MyMethod、额外训练种子或顺序语言训练，旧mCoT路线B保持用户停止。\n\n公开材料包含科学代码、配置、来源ID/输入指纹、评分统计及配对区间；不包含权重、原始训练文本、原始新闻文章或内部服务器运行管理信息。6400条数据已通过按公开来源ID独立重建并逐输入/标签指纹比对，全部7个冻结训练文件也逐字节一致。运行源码和配置以实际SHA为准；GitHub交付快照包含部署路径适配，不与日志中的旧checkout HEAD混同。\n\n## 图与入口\n\n![目标任务成绩](../../results/aya_4b_20261105/analysis-v1/figures/target_scores.png)\n\n![通用能力变化](../../results/aya_4b_20261105/analysis-v1/figures/general_retention.png)\n\n- [逐任务结果和配对区间](results.md)\n- [机器可读结果](../../results/aya_4b_20261105/analysis-v1/scores.csv)\n- [配对比较](../../results/aya_4b_20261105/analysis-v1/paired_comparisons.csv)\n- [输入重建身份](../../results/aya_4b_20261105/reproduction-inputs-v1/reproduction_spec.json)\n'
    report = report.replace('第二个LR2e-5候选NOT_RUN。', '第二个预定LR2e-5候选已启动；其最新状态和最终结果见独立候选报告，本表不据此改写。')
    write(a.report / 'final_report.md', report)
    write(a.report / 'chatgpt_handoff.md', '# 交给ChatGPT：Qwen3-4B SW适配\n\n原四模型固定预算实验和完整评测已完成；翻译修复仍未确认。先阅读[完整报告](final_report.md)和[逐任务表](results.md)。\n\n' + '\n'.join(absolute) + '\n\n讨论重点：目标收益是否跨阅读、知识、摘要、翻译一致；通用损失集中在哪些任务；SSU是否在本轮配方下体现收益—保留取舍。模型规模和数据同时改变，不把与旧1.7B的差异单独归因于规模。只有seed42；尚无新方法，未开展顺序多语言实验。旧路线B仍停止。\n')
    supplement = a.analysis.parent / 'supplemental-analysis-v1/scores.json'
    if supplement.exists():
        extra = json.loads(supplement.read_text())
        values = {(x['model'], x['task']): x['score'] for x in extra}
        table = ['\n\n## 补充全量 benchmark：XNLI 与 XCOPA', '', '指标为0-shot指令选择题likelihood ACC（%）；XNLI EN/SW各5010题，XCOPA SW500题。训练后冻结的探索性补充，不是原80项主辅指标的一部分。', '', '|任务|Base|FFT|LoRA|SSU-Wanda|', '|---|---:|---:|---:|---:|']
        for task, label in [('xnli_sw', 'XNLI SW'), ('xnli_en', 'XNLI EN'), ('xcopa_sw', 'XCOPA SW')]:
            table.append('|' + label + '|' + '|'.join((f'{values[role, task]:.2f}' for role in ROLES)) + '|')
        table += ['', 'XNLI有明显提示/标签偏好敏感性；不能将全部零样本增益解释为纯语义学习。192题/语言的3-shot开发对照不覆盖全量表。XCOPA提升较小且方法区间不同；详见supplemental_benchmarks.md和xnli_format_diagnosis.md。主线加补充共92项分数、69项配对比较，开发诊断另列。']
        for name in ['final_report.md', 'chatgpt_handoff.md']:
            target = a.report / name
            target.write_text(target.read_text() + '\n'.join(table) + '\n')
    print(json.dumps({'status': 'FINAL_REPORT_RENDERED', 'main_metrics': len(MAIN), 'score_rows': len(scores), 'paired_comparisons': len(paired), 'figures': 4, 'report': str(a.report)}, ensure_ascii=False))
if __name__ == '__main__':
    main()
