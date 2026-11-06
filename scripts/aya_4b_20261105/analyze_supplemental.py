"""Audit every supplemental choice score and paired change, including label bias."""
from profile_paths import resolve
import json, csv, gzip, hashlib, argparse, collections, math, sys, datetime
from pathlib import Path
import numpy as np
R = Path(resolve('ROOT'))
sys.path.insert(0, str(R / 'scripts/aya_4b_20261105'))
from analyze import paired_ratio
L = Path(resolve('LOGS') + '/supplemental-benchmarks-v1')
D = Path(resolve('DATA') + '/supplemental-v1')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def hs(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def dump(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--partial', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    I = a.output / 'per_item_scores'
    I.mkdir()
    values = {}
    scores = []
    bias = []
    identities = {}
    manifest = []
    pending = []
    cost = []
    protocol = json.loads((D / 'manifest.json').read_text())
    expected = {x['task']: x for x in protocol['tasks']}
    for method in ['base', 'lora', 'fft', 'ssu']:
        root = L / method
        if not (root / 'DONE.json').exists():
            pending.append(method)
            continue
        files = list((root / 'results').rglob('results_*.json'))
        assert len(files) == 1
        result = json.loads(files[0].read_text())
        actual = result['config']['model_args']
        expected_model = resolve('MODEL') if method in ['base', 'lora'] else resolve('CHECKPOINTS') + '/' + method + '_final_v2'
        assert actual['pretrained'] == expected_model and actual['tokenizer'] == resolve('MODEL') and (actual['enable_thinking'] is False)
        if method == 'lora':
            assert actual['peft'] == resolve('CHECKPOINTS') + '/lora_final_v1'
        else:
            assert not actual.get('peft')
        assert set(result['results']) == {'aya4b_' + x for x in expected}
        assert all((v['num_fewshot'] == 0 for v in result['configs'].values()))
        identities[method + '/results.json'] = sha(files[0])
        cost.append({'model': method, 'total_evaluation_time_seconds': result['total_evaluation_time_seconds']})
        for task, spec in expected.items():
            name = 'aya4b_' + task
            samplefiles = list(files[0].parent.glob('samples_' + name + '_*.jsonl'))
            assert len(samplefiles) == 1
            samplefile = samplefiles[0]
            samples = [json.loads(x) for x in samplefile.open()]
            sourcefile = D / spec['splits']['test']['file']
            assert sha(sourcefile) == spec['splits']['test']['sha256']
            source = {x['id']: x for x in map(json.loads, sourcefile.open())}
            assert len(samples) == len(source) == spec['splits']['test']['rows']
            items = {}
            matrix = np.zeros((spec['options'], spec['options']), dtype=int)
            predicted = collections.Counter()
            goldcounts = collections.Counter()
            for row in samples:
                doc = row['doc']
                k = doc['id']
                assert k not in items and doc == source[k]
                assert row['filter'] == 'none'
                ll = [float(x[0]) for x in row['filtered_resps']]
                assert len(ll) == spec['options'] and all((math.isfinite(v) for v in ll))
                gold = int(doc['label'])
                assert int(row['target']) == gold
                pred = max(range(len(ll)), key=ll.__getitem__)
                correct = int(pred == gold)
                assert correct == row['acc']
                goldcounts[gold] += 1
                predicted[pred] += 1
                matrix[gold, pred] += 1
                items[k] = {'identity': hs([doc['prompt'], gold]), 'cluster': doc['premise_cluster_sha256'] if task.startswith('xnli') else k, 'numerator': correct, 'denominator': 1, 'gold_index': gold, 'predicted_index': pred}
            actual = 100 * sum((x['numerator'] for x in items.values())) / len(items)
            reported = result['results'][name]['acc,none'] * 100
            assert abs(actual - reported) < 1e-10
            values[method, task] = items
            scores.append({'model': method, 'task': task, 'metric': 'ACC', 'score': actual, 'unit': 'percent', 'samples': len(items), 'status': 'VERIFIED_COMPLETE'})
            bias.append({'model': method, 'task': task, 'predicted_label_counts': dict(predicted), 'gold_label_counts': dict(goldcounts), 'confusion_matrix_rows_gold_columns_predicted': matrix.tolist(), 'majority_gold_baseline_ACC': 100 * max(goldcounts.values()) / len(items), 'largest_prediction_class_fraction': max(predicted.values()) / len(items), 'use': 'Choice preference and task-format diagnosis only; not a replacement for main ACC'})
            identities[method + '/' + samplefile.name] = sha(samplefile)
            dest = I / (method + '_' + task + '.csv.gz')
            with gzip.open(dest, 'wt', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['item_id_sha256', 'identity_sha256', 'cluster_sha256', 'numerator', 'denominator'])
                writer.writeheader()
                for k, v in sorted(items.items()):
                    writer.writerow({'item_id_sha256': hs(k), 'identity_sha256': v['identity'], 'cluster_sha256': hs(v['cluster']), 'numerator': v['numerator'], 'denominator': 1})
            manifest.append({'model': method, 'task': task, 'metric': 'ACC', 'file': dest.name, 'sha256': sha(dest), 'rows': len(items)})
    comparisons = []
    for (method, task), items in values.items():
        if method == 'base' or ('base', task) not in values:
            continue
        comparisons.append({'model': method, 'task': task, 'metric': 'ACC', **paired_ratio(items, values['base', task])})
    if not a.partial:
        assert not pending and len(scores) == 12 and (len(comparisons) == 9)
    (a.output / 'corpus_statistics').mkdir()
    dump(a.output / 'corpus_statistics/manifest.json', [])
    dump(a.output / 'scores.json', scores)
    dump(a.output / 'paired_comparisons.json', comparisons)
    dump(a.output / 'label_bias_diagnostics.json', bias)
    dump(a.output / 'source_identity.json', identities)
    dump(I / 'manifest.json', manifest)
    dump(a.output / 'cost.json', {'formal_evaluations': cost, 'engineering_smoke_is_validation_only': True, 'scope': 'Supplemental tasks, not included in original32evaluationphase costs'})
    dump(a.output / 'protocol.json', protocol)
    dump(a.output / 'DONE.json', {'status': 'PARTIAL' if a.partial else 'VERIFIED_ALL_FIXED_MODELS', 'scores': len(scores), 'paired_comparisons': len(comparisons), 'pending_models': pending, 'raw_text_excluded': True, 'bootstrap': '1000 repeats seed42, NLI shared-premise clusters; exploratory conditional intervals, no multiseed stability'})
    lines = ['# 多语言补充评测：固定模型结果', '', '这是训练完成后、按用户新目标增加的探索性补充比较。三项任务在首次新增评分前固定，但不是训练前预登记的主要终点；没有依据补充分数重新选择权重、提示或删去任务。', '', '统一0-shot、固定英文任务说明、A/B/C选项likelihood ACC、非思考、batch4、seed42。XNLI使用指令选择题格式，与Harness原始无指令completion模板不同。所有模型使用相同协议，不能与其他论文默认模板分数直接等同。', '', '|任务|Base|LoRA|FFT|SSU-Wanda|', '|---|---:|---:|---:|---:|']
    for task in expected:
        cols = []
        for role in ['base', 'lora', 'fft', 'ssu']:
            row = next((x for x in scores if x['model'] == role and x['task'] == task), None)
            cols.append('待完成' if row is None else f"{row['score']:.2f}")
        lines.append('|' + task + '|' + '|'.join(cols) + '|')
    lines += ['', '逐题正确率已由保存的选项likelihood和原始gold重算；对齐全部实际题目身份、数目和提示。标签偏好诊断见JSON：若Base集中选择单一标签，应限定解释为该任务及输出形式的综合表现变化，不能仅凭ACC改善声称获得了纯语义推理能力。', '', 'XNLI官方test每语言5010题、XCOPA SW500题；后者无XCOPA-en配置，未伪造英语对应分数。现有英语阅读/知识/数学/指令任务继续保留。', '', '本表未按分数筛除任务；翻译与Global-MMLU原结果仍独立保留。配对95%区间按XNLI共享前提聚类，未修正多重比较，只针对当前seed42权重。', '', '各模型任务的真实耗时见cost.json；额外运行量单独记录。']
    (a.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'score_rows': len(scores), 'paired_comparisons': len(comparisons), 'pending': pending}))
if __name__ == '__main__':
    main()
