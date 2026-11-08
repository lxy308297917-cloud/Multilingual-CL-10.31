"""Human tables and scientific figures from fully audited endpoints only."""
from pathlib import Path
import os
import collections
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[3]
O = R / 'research/functional_representation_20261109/p2'
K = Path(os.environ['P2_CHECKPOINT_ROOT'])
MODELS = ['Base', 'LegacyFull', 'all_layer', 'full_feature', 'ce_replay']
LABELS = {'Base': '适配前模型', 'LegacyFull': '旧Full LoRA（batch1，历史）', 'all_layer': '新All-layer LoRA（batch2）',
          'full_feature': '普通全表示保持', 'ce_replay': '同池英语CE回放'}
ENGLISH = {'all_layer': 'All-layer LoRA', 'full_feature': 'Full-feature preservation', 'ce_replay': 'Same-pool CE replay'}
MAIN = [('Belebele target', 'ACC'), ('XL-Sum target', 'ROUGE-Lsum'), ('Belebele EN', 'ACC'),
        ('Global-MMLU SW', 'ACC'), ('GSM8K', 'flexible-extract'), ('IFEval', 'prompt_level_strict_acc')]

def read(p):
    return list(csv.DictReader(p.open()))

def main():
    for file in ['collection_verification.json', 'training_artifact_verification.json',
                 'runtime_protocol_verification.json', 'numeric_recomputation_verification.json', 'cost_summary.json']:
        assert json.loads((O / file).read_text())['complete'], file
    scores = read(O / 'scores.csv');pairs = read(O / 'paired_changes.csv')
    assert len(scores) == 40 and len(pairs) == 48
    values = {(x['model'], x['task'], x['metric']): float(x['score']) for x in scores}
    changes = {(x['model'], x['reference'], x['task'], x['metric']): x for x in pairs}
    text = ['# P2全部正式端点结果', '',
            '所有新模型均为同目标6400条、400步Final。三项新对照共同batch2；旧batch1 Full只作历史旁证。除ROUGE-Lsum外均为百分比，越高越好。', '',
            '|模型|SW阅读ACC|SW摘要ROUGE-Lsum|EN阅读ACC|SW Global-MMLU ACC|GSM8K flexible|IFEval prompt strict|',
            '|---|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        text.append('|' + LABELS[model] + '|' + '|'.join(f'{values[(model, *m)]:.2f}' for m in MAIN) + '|')
    text += ['', '## 严格提取与两级指令指标', '',
             '|模型|GSM8K strict|GSM8K flexible|IFEval prompt strict|IFEval instruction strict|',
             '|---|---:|---:|---:|---:|']
    secondary = [('GSM8K', 'strict-match'), ('GSM8K', 'flexible-extract'), ('IFEval', 'prompt_level_strict_acc'), ('IFEval', 'inst_level_strict_acc')]
    for model in MODELS:
        text.append('|' + LABELS[model] + '|' + '|'.join(f'{values[(model, *m)]:.2f}' for m in secondary) + '|')
    text += ['', '## 48项配对变化', '',
             '差值为模型减参考；阅读、知识、数学、指令为百分点，摘要为ROUGE分。2000次配对bootstrap；指令级按提示聚类。单训练seed，未多重校正，未设非劣效容忍度；区间跨零不证明等效。', '',
             '|模型|参考|任务／指标|变化|配对95%区间|', '|---|---|---|---:|---|']
    for x in pairs:
        text.append(f"|{LABELS[x['model']]}|{LABELS[x['reference']]}|{x['task']}／{x['metric']}|{float(x['difference']):+.2f}|[{float(x['ci95_low']):+.2f}, {float(x['ci95_high']):+.2f}]|")
    text += ['', '## 复算与边界', '',
             '- scores.csv包含40条绝对分数，paired_changes.csv包含48条配对比较；所有正负项均保留。',
             '- numeric_items.json仅含匿名顺序下的逐题数值、权重和身份摘要，可在CPU复算，不含原题、回答或模型。',
             '- 285条学科分解见gmmlu_subject_scores.csv；总体知识ACC按14042题加权，不是57学科宏平均。',
             '- IFEval类型统计是描述性分解，不用于选方法；GSM8K严格／灵活提取差异包含输出格式因素，不自动等同纯推理差异。',
             '- 仅SW、单阶段适配、单seed42，尚非顺序CL或跨语言方法泛化的证据；所有中间节点不参与Final选择。',
             '- 本轮未重新测翻译或XNLI，历史结果保留，不能声称其问题已解决。', '']
    (O / 'results.md').write_text('\n'.join(text))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.flat, MAIN):
        ax.axhline(0, color='gray', linewidth=.8)
        for i, model in enumerate(['full_feature', 'ce_replay']):
            x = changes[(model, 'all_layer', *metric)]
            value, lo, hi = [float(x[k]) for k in ['difference', 'ci95_low', 'ci95_high']]
            ax.errorbar(i, value, yerr=[[max(0, value - lo)], [max(0, hi - value)]], fmt='o', capsize=4, color=['tab:blue', 'tab:orange'][i])
        ax.set_xticks([0, 1], ['Full feature', 'CE replay']);ax.set_xlim(-.5, 1.5)
        ax.set_title(metric[0] + '\n' + metric[1], fontsize=10)
        ax.set_ylabel('ROUGE-Lsum points' if metric[1] == 'ROUGE-Lsum' else 'Percentage points')
    fig.suptitle('Final400 minus fresh all-layer LoRA: paired95% intervals, single seed42')
    fig.tight_layout();fig.savefig(O / 'final_tradeoffs.png', dpi=180);plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    curve_rows = []
    for model in ['all_layer', 'full_feature', 'ce_replay']:
        root = K / (model + '_final_v1')
        history = [x for x in json.loads((root / 'trainer_state.json').read_text())['log_history'] if 'loss' in x]
        if model == 'all_layer':
            xs = [x['step'] for x in history];ys = [x['loss'] for x in history]
            axes[0].plot(xs, ys, label=ENGLISH[model])
            curve_rows += [dict(model=model, step=x, target_ce=y, old_objective=None) for x, y in zip(xs, ys)]
        else:
            trace = [json.loads(x) for x in (root / 'old_exposure.jsonl').open()]
            target, old = collections.defaultdict(float), collections.defaultdict(list)
            for x in trace:
                target[x['step']] += x['target_loss_contribution'];old[x['step']].append(x['old_objective'])
            xs, ys, zs = [], [], []
            for x in history:
                step = x['step'];window = range(step - 4, step + 1)
                y = float(np.mean([target[k] for k in window]));z = float(np.mean([np.mean(old[k]) for k in window]))
                assert abs(y + z - x['loss']) < 1e-3, (model, step, 'Combined loss decomposition differs')
                xs.append(step);ys.append(y);zs.append(z)
                curve_rows.append(dict(model=model, step=step, target_ce=y, old_objective=z))
            axes[0].plot(xs, ys, label=ENGLISH[model]);axes[1 if model == 'full_feature' else 2].plot(xs, zs)
    axes[0].set_title('Target SFT CE (five-step means)');axes[0].legend(fontsize=8)
    axes[1].set_title('Full-feature normalized penalty');axes[2].set_title('CE replay old assistant CE')
    for ax in axes:ax.set_xlabel('Optimizer step');ax.grid(alpha=.2)
    fig.suptitle('Training diagnostics only; no benchmark selection or early stopping')
    fig.tight_layout();fig.savefig(O / 'training_objectives.png', dpi=180);plt.close(fig)
    with (O / 'training_objective_curves.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0]));writer.writeheader();writer.writerows(curve_rows)
    print('Complete tables, final tradeoff and diagnostic objective figures generated.')

if __name__ == '__main__':
    main()
