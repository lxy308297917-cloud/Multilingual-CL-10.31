"""Audit a small development-only intervention on demonstration count."""
from profile_paths import resolve
import argparse, json, csv, gzip, hashlib, collections, math, sys
from pathlib import Path
R = Path(resolve('ROOT'))
sys.path.insert(0, str(R / 'scripts/aya_4b_20261105'))
from analyze import paired_ratio

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def hs(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert (a.run / 'DONE.json').exists()
    a.output.mkdir(exist_ok=False)
    protocol = json.loads((a.data / 'manifest.json').read_text())
    assert protocol == json.loads((a.run / 'protocol.json').read_text())
    scores = []
    values = {}
    cost = []
    items_manifest = []
    itemdir = a.output / 'per_item_scores'
    itemdir.mkdir()
    for model in ['base', 'fft']:
        rfiles = list((a.run / model / 'results').rglob('results_*.json'))
        assert len(rfiles) == 1
        r = json.loads(rfiles[0].read_text())
        cost.append({'model': model, 'total_evaluation_time_seconds': r['total_evaluation_time_seconds'], 'source_results_sha256': sha(rfiles[0])})
        expected_model = resolve('MODEL') if model == 'base' else resolve('CHECKPOINTS') + '/fft_final_v2'
        assert r['config']['model_args']['pretrained'] == expected_model
        for lang in ['sw', 'en']:
            src = {x['id']: x for x in map(json.loads, (a.data / (lang + '_development192.jsonl')).open())}
            demos = list(map(json.loads, (a.data / (lang + '_demonstrations3.jsonl')).open()))
            assert len(src) == 192 and len(demos) == 3 and (not set(src) & {x['id'] for x in demos})
            for shots in [0, 3]:
                name = 'aya4b_xnli_' + lang + '_format_dev_' + str(shots) + 'shot'
                assert r['configs'][name]['num_fewshot'] == shots
                samples = list(rfiles[0].parent.glob('samples_' + name + '_*.jsonl'))
                assert len(samples) == 1
                rows = list(map(json.loads, samples[0].open()))
                assert len(rows) == 192
                items = {}
                preds = collections.Counter()
                goldcounts = collections.Counter()
                for x in rows:
                    doc = x['doc']
                    key = doc['id']
                    assert doc == src[key] and key not in items
                    ll = [float(v[0]) for v in x['filtered_resps']]
                    assert len(ll) == 3 and all((math.isfinite(v) for v in ll))
                    gold = int(x['target'])
                    assert gold == doc['label']
                    pred = max(range(3), key=ll.__getitem__)
                    correct = int(pred == gold)
                    assert correct == x['acc']
                    context = next(iter(x['arguments'].values()))['arg_0']
                    assert all(((d['prompt'] in context) == bool(shots) for d in demos))
                    preds[pred] += 1
                    goldcounts[gold] += 1
                    items[key] = {'identity': hs([doc['prompt'], doc['label']]), 'cluster': doc['premise_cluster_sha256'], 'numerator': correct, 'denominator': 1}
                assert dict(goldcounts) == {0: 64, 1: 64, 2: 64}
                actual = 100 * sum((x['numerator'] for x in items.values())) / 192
                assert abs(actual - 100 * r['results'][name]['acc,none']) < 1e-10
                values[model, lang, shots] = items
                scores.append({'model': model, 'language': lang, 'shots': shots, 'samples': 192, 'ACC': actual, 'predicted_label_counts': dict(preds), 'largest_class_fraction': max(preds.values()) / 192, 'actual_demonstrations_applied_and_disjoint_verified': True, 'sample_file_sha256': sha(samples[0])})
                path = itemdir / (model + '_' + lang + '_' + str(shots) + 'shot.csv.gz')
                with gzip.open(path, 'wt', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['item_id_sha256', 'identity_sha256', 'cluster_sha256', 'numerator', 'denominator'])
                    w.writeheader()
                    for k, v in sorted(items.items()):
                        w.writerow({'item_id_sha256': hs(k), 'identity_sha256': v['identity'], 'cluster_sha256': hs(v['cluster']), 'numerator': v['numerator'], 'denominator': 1})
                items_manifest.append({'model': model, 'language': lang, 'shots': shots, 'file': path.name, 'sha256': sha(path)})
    comparisons = []
    for lang in ['sw', 'en']:
        for shots in [0, 3]:
            comparisons.append({'contrast': 'FFT_minus_Base', 'language': lang, 'shots': shots, **paired_ratio(values['fft', lang, shots], values['base', lang, shots])})
        for model in ['base', 'fft']:
            comparisons.append({'contrast': '3shot_minus_0shot', 'language': lang, 'model': model, **paired_ratio(values[model, lang, 3], values[model, lang, 0])})
    dump(a.output / 'results.json', {'status': 'VERIFIED_DEVELOPMENT_ONLY', 'scores': scores, 'paired_comparisons': comparisons, 'protocol': protocol, 'cost': cost, 'raw_text_excluded': True, 'caveats': ['One small balanced development sample and one fixed demonstration set; no selected best prompt or test-score replacement', 'Not cross-seed or cross-demonstration robustness', 'Shared-premise clustered unadjusted95% intervals,1000bootstrap seed42']})
    dump(itemdir / 'manifest.json', items_manifest)
    lines = ['# XNLI示例数量开发对照', '', '仅使用validation中192题/语言（每类64题），Base与FFT固定权重，0与3个类别均衡的示例。示例和目标不重叠，所有对照列保留。', '', '|模型|语言|示例数|ACC%|最大预测类别占比|', '|---|---|---:|---:|---:|']
    for x in scores:
        lines.append(f"|{x['model']}|{x['language']}|{x['shots']}|{x['ACC']:.2f}|{100 * x['largest_class_fraction']:.1f}%|")
    lines += ['', '|对比|语言|设置|ACC变化pp|配对95%区间|', '|---|---|---|---:|---|']
    for x in comparisons:
        lines.append(f"|{x['contrast']}|{x['language']}|{x.get('shots', x.get('model'))}|{x['delta']:+.2f}|[{x['ci95_low']:+.2f}, {x['ci95_high']:+.2f}]|")
    lines += ['', '这是对零样本首字母偏好风险的开发诊断，不覆盖原5010题正式补充表。示例数量同时改变了格式示范和任务示范，不能将差异归因于单一机制。是否将XNLI作为后续主要任务，应结合两套开发结果和原有阅读/摘要/翻译/知识/通用矩阵决定，不选择最高列冒充固定协议。']
    (a.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'VERIFIED_DEVELOPMENT_ONLY', 'score_rows': len(scores), 'paired_intervals': len(comparisons)}))
if __name__ == '__main__':
    main()
