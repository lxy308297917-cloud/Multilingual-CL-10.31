"""Audit the preregistered development comparison and export text-free corpus statistics."""
from profile_paths import resolve
import argparse, json, hashlib, collections, csv, gzip, sys
from pathlib import Path
import numpy as np
from sacrebleu.metrics import CHRF, BLEU
R = Path(resolve('ROOT'))
sys.path.insert(0, str(R / 'scripts/bench_v2'))
from paired_statistics import paired_corpus

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def hs(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert (a.run / 'DONE.json').exists()
    a.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((a.run / 'summary.json').read_text())
    rows = [json.loads(x) for x in (a.run / 'predictions.jsonl').open()]
    assert len(rows) == 512 and summary['predictions_sha256'] == sha(a.run / 'predictions.jsonl')
    groups = collections.defaultdict(list)
    for x in rows:
        groups[x['model'], x['profile'], x['direction']].append(x)
    assert len(groups) == 8 and all((len(xs) == len({x['id'] for x in xs}) == 64 for xs in groups.values()))
    stats = []
    for (model, profile, direction), xs in groups.items():
        reported = next((x for x in summary['rows'] if (x['model'], x['profile'], x['direction']) == (model, profile, direction)))
        for metric, scorer in [('chrF++', CHRF(word_order=2)), ('BLEU', BLEU(tokenize='13a'))]:
            matrix = np.asarray(scorer._extract_corpus_statistics([x['prediction'] for x in xs], [[x['reference'] for x in xs]]), dtype=np.int64)
            score = scorer._compute_score_from_stats(matrix.sum(axis=0).tolist()).score
            assert abs(score - reported[metric]) < 1e-10
            dest = a.output / (model + '_' + profile + '_' + direction.replace('->', '_') + '_' + metric + '.csv.gz')
            with gzip.open(dest, 'wt', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['item_id_sha256', 'identity_sha256', 'statistics'])
                w.writeheader()
                for x, v in zip(xs, matrix):
                    w.writerow({'item_id_sha256': hs(x['id']), 'identity_sha256': hs([x['source'], x['reference']]), 'statistics': json.dumps(v.tolist(), separators=(',', ':'))})
            stats.append({'model': model, 'profile': profile, 'direction': direction, 'metric': metric, 'score': score, 'samples': 64, 'file': dest.name, 'sha256': sha(dest)})
    comparisons = []

    def compare(label, target, base):
        xs, bs = (groups[target], groups[base])
        assert [(x['id'], x['source'], x['reference']) for x in xs] == [(x['id'], x['source'], x['reference']) for x in bs]
        for metric in ['chrF++', 'BLEU']:
            ci = paired_corpus([x['prediction'] for x in xs], [x['prediction'] for x in bs], [x['reference'] for x in xs], metric=metric)
            comparisons.append({'contrast': label, 'target': list(target), 'base': list(base), 'metric': metric, **ci})
    for direction in ['en->sw', 'sw->en']:
        for model in ['base', 'lora']:
            compare('official_sampling_minus_greedy', (model, 'official_non_thinking', direction), (model, 'greedy', direction))
        for profile in ['greedy', 'official_non_thinking']:
            compare('lora_minus_base', ('lora', profile, direction), ('base', profile, direction))
    result = {'status': 'VERIFIED_DEVELOPMENT_COMPARISON_NOT_FORMAL_TEST', 'scores': summary['rows'], 'paired_comparisons': comparisons, 'corpus_statistics': stats, 'source_predictions_sha256': sha(a.run / 'predictions.jsonl'), 'frozen_manifest_sha256': sha(a.run / 'manifest.json'), 'cost': json.loads((a.run / 'cost.json').read_text()), 'limitations': ['64paireddevexamples per direction; historically exposed public corpus', 'One fixed stochastic inference seed; intervals conditional on observed generations', 'Base/no-adapter and LoRA share an enable_lora engine, separate numerical profile from formal evaluation', 'No formal predictions, scores, masks or checkpoints modified']}
    (a.output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 翻译解码开发对照', '', '固定64个FLORES dev句组，两方向；Base与LoRA Final400；原greedy和官方non-thinking采样两套预定配置。排除原3个few-shot句组。仅为开发诊断，不覆盖正式表。', '', '|模型|解码|方向|chrF++|512上限条数/64|', '|---|---|---|---:|---:|']
    for x in summary['rows']:
        lines.append(f"|{x['model']}|{x['profile']}|{x['direction']}|{x['chrF++']:.2f}|{x['cap_count']}|")
    lines += ['', '## 配对差异', '', '|对比|模型或解码|方向|chrF++差异|配对95%区间|', '|---|---|---|---:|---|']
    for x in comparisons:
        if x['metric'] != 'chrF++':
            continue
        label = x['target'][0] if x['contrast'] == 'official_sampling_minus_greedy' else x['target'][1]
        lines.append(f"|{x['contrast']}|{label}|{x['target'][2]}|{x['delta']:+.2f}|[{x['ci95_low']:+.2f}, {x['ci95_high']:+.2f}]|")
    official_comparisons = [x for x in comparisons if x['contrast'] == 'lora_minus_base' and x['target'][1] == 'official_non_thinking' and (x['metric'] == 'chrF++')]
    lines += ['', '## 当前权重与Base的开发差异（官方non-thinking采样）']
    for x in official_comparisons:
        lines.append(f"- {x['target'][2]}：chrF++差异{x['delta']:+.2f}，配对95%区间[{x['ci95_low']:+.2f}, {x['ci95_high']:+.2f}]。")
    lines += ['', '重复、长度上限与分数变化按各组真实统计解释。此处只验证当前固定权重和预定义解码；不把采样改善直接解释为训练学会了翻译，也不覆盖正式表。', '', '有限数据修正只依据冻结开发规则推进，正式test不挑配置。', '', '区间为1000次配对bootstrap、seed42、未修正多重比较；只针对当前观察到的输出，不代表跨推理/训练种子稳定性。', '', '官方采样参数来源：https://huggingface.co/Qwen/Qwen3-4B#best-practices 。']
    (a.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': result['status'], 'score_groups': 8, 'paired_comparisons': len(comparisons), 'corpus_statistic_files': len(stats)}))
if __name__ == '__main__':
    main()
