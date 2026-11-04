"""Standalone scientific figures from aggregate JSON; no model evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = ['W1', 'A', 'B', 'C', 'D']
NAMES = ['SW start', 'SFT', 'Marginal', 'Content KD', 'Replay']
COLORS = ['#777777', '#3876ad', '#d28b26', '#31826a', '#96538e']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', type=Path, required=True)
    parser.add_argument('--formal', type=Path, required=True)
    parser.add_argument('--p0', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fresh = json.loads(args.fresh.read_text())
    formal = json.loads(args.formal.read_text())
    p0 = json.loads(args.p0.read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    protocols = [('raw_letter', 'Original letter'),
                 ('rotation_content_average', 'Four-rotation average'),
                 ('content_mean', 'Content token mean')]
    x = np.arange(len(LABELS))
    for ax, lang in zip(axes, ['en', 'sw', 'ha']):
        for j, (metric, name) in enumerate(protocols):
            values = [100 * fresh['summaries'][m][lang]['metrics'][metric]['mean']
                      for m in LABELS]
            ax.bar(x + (j - 1) * .24, values, width=.23, label=name)
        ax.set_xticks(x, NAMES, rotation=25, ha='right')
        ax.set_title(lang.upper() + ': fresh science QA, n=120')
        ax.set_ylim(0, 100)
        ax.axhline(25, color='gray', linestyle=':', linewidth=.8)
    axes[0].set_ylabel('Accuracy (%)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .89))
    fig.savefig(args.output / 'fresh_holdout_readouts.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    comparisons = [
        ('HA Belebele|acc|none', 'SW Belebele|acc|none', 'HA reading', 'SW reading'),
        ('HA Global-MMLU|acc|none', 'EN MMLU|acc|none', 'HA knowledge', 'EN knowledge'),
        ('HA Belebele|acc|none', 'sft_v1_ifeval|prompt_level_strict_acc|none',
         'HA reading', 'EN IFEval strict')]
    for ax, (kx, ky, nx, ny) in zip(axes, comparisons):
        for method, name, color in zip(LABELS[1:], NAMES[1:], COLORS[1:]):
            contrast = formal['contrasts'][method + '_minus_W1']
            xx, yy = contrast[kx], contrast[ky]
            point = np.array([xx['delta'], yy['delta']]) * 100
            xerr = np.array([[max(0, xx['delta'] - xx['low'])],
                             [max(0, xx['high'] - xx['delta'])]]) * 100
            yerr = np.array([[max(0, yy['delta'] - yy['low'])],
                             [max(0, yy['high'] - yy['delta'])]]) * 100
            ax.errorbar(*point, xerr=xerr, yerr=yerr, fmt='o', color=color,
                        capsize=2, alpha=.85, label=name)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.axvline(0, color='gray', linewidth=.8)
        ax.plot(0, 0, marker='+', color='black')
        ax.set_xlabel(nx + ' change from W1 (pp)')
        ax.set_ylabel(ny + ' change from W1 (pp)')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Fixed100 pilot: target learning and old-task change\n'
                 'Paired 95% intervals; one seed; no multiplicity adjustment', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output / 'learning_and_retention.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, model, title in zip(axes, ['W1', 'A100'], ['SW start', 'After ordinary HA SFT100']):
        xx = np.arange(3)
        for j, (field, label) in enumerate([
            ('token_counts_crossed16', 'Token A'),
            ('physical_position_counts_crossed16', 'First physical position')]):
            values = []
            for lang in ['en', 'sw', 'ha']:
                counts = p0['summaries'][model][lang][field]
                values.append(100 * counts[0] / sum(counts))
            ax.bar(xx + (j - .5) * .32, values, width=.31, label=label)
        ax.set_xticks(xx, ['EN', 'SW', 'HA'])
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.axhline(25, color='gray', linestyle=':', linewidth=.8)
    axes[0].set_ylabel('Prediction share across crossed16 prompts (%)')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / 'token_vs_physical_position.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
