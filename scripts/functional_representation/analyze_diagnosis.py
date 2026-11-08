"""Frozen exploratory held-out comparison; proxies never replace benchmark efficacy."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

import os
R = Path(__file__).resolve().parents[2]
DATA = Path(os.environ['FUNCTIONAL_DIAGNOSTIC_DATA'])
BEHAVIOR = Path(os.environ['FUNCTIONAL_BEHAVIOR_LOG'])
O = R / 'research/functional_representation_20261109'
MODELS = ['full', 'remove_middle', 'uniform']


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    differences = positive[:, None] - negative[None, :]
    return float((differences > 0).mean() + .5 * (differences == 0).mean())


def main():
    protocol_path = R / 'configs/functional_representation_20261109/analysis_plan.json'
    protocol = load(protocol_path)
    assert not protocol['preregistered'] and protocol['uncertainty']['resamples'] == 2000
    behavior_report = load(BEHAVIOR / 'DONE.json')
    representation_root = DATA / 'variant_measurements_v1'
    assert load(representation_root / 'DONE.json')['per_example_rows'] == 768
    manifest = load(DATA / 'manifest.json')
    control = manifest['control']
    norm = {'full': 1., 'remove_middle': control['remove_middle_delta_squared_norm'] / control['full_delta_squared_norm'],
            'uniform': control['actual_uniform_delta_squared_norm'] / control['full_delta_squared_norm']}
    meta = load(DATA / 'base_features_v1/rows.json')
    baseline = {x['id']: x for x in meta}
    behavior, feature_rows = {}, []
    for name in ['base'] + MODELS:
        report = load(BEHAVIOR / name / 'DONE.json')
        path = BEHAVIOR / name / 'predictions.jsonl'
        assert sha(path) == report['predictions_sha256']
        rows = [json.loads(v) for v in path.open()]
        assert len(rows) == 256 and len({x['id'] for x in rows}) == 256
        for row in rows:
            reference = baseline[row['id']]
            assert row['split'] == reference['split'] and row['capability'] == reference['capability']
            behavior[(name, row['id'])] = bool(row['correct'])
    for name in MODELS:
        path = representation_root / name / 'per_example.csv'
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 256 and len({x['id'] for x in rows}) == 256
        for x in rows:
            row = {**x, **{k: float(x[k]) for k in ['assistant_ce', 'base_ce', 'ce_delta', 'answer_total', 'answer_functional', 'answer_random', 'question_total']}}
            row['effective_delta_squared_norm_ratio'] = norm[name]
            assert abs(row['base_ce'] - baseline[row['id']]['assistant_ce']) < 1e-8
            assert abs(row['ce_delta'] - (row['assistant_ce'] - row['base_ce'])) < 1e-8
            feature_rows.append(row)
    assert len(feature_rows) == len(behavior) * 3 // 4 == 768
    tables, comparisons, predictions, fits, gates = [], [], [], {}, []
    rng = np.random.default_rng(42)
    for cap in ['math', 'instruction']:
        train = sorted([x for x in feature_rows if x['capability'] == cap and x['split'] == 'build'], key=lambda x: (x['id'], x['variant']))
        test = sorted([x for x in feature_rows if x['capability'] == cap and x['split'] == 'check'], key=lambda x: (x['id'], x['variant']))
        assert len(train) == len(test) == 192
        assert {x['id'] for x in train}.isdisjoint({x['id'] for x in test})
        ytrain = np.array([x['ce_delta'] for x in train])
        ytest = np.array([x['ce_delta'] for x in test])
        labels = np.array([int(not behavior[(x['variant'], x['id'])]) for x in test])
        residuals, aucs = {}, {}
        for name, fields in protocol['features'].items():
            xtrain = np.array([[x[k] for k in fields] for x in train], dtype=np.float64)
            xtest = np.array([[x[k] for k in fields] for x in test], dtype=np.float64)
            center, scale = xtrain.mean(axis=0), xtrain.std(axis=0)
            scale[scale < 1e-12] = 1.
            design = np.column_stack([np.ones(len(train)), (xtrain - center) / scale])
            beta = np.linalg.lstsq(design, ytrain, rcond=None)[0]
            predicted = np.column_stack([np.ones(len(test)), (xtest - center) / scale]) @ beta
            squared_errors = (predicted - ytest) ** 2
            residuals[name] = squared_errors
            mse = float(squared_errors.mean())
            denominator = float(((ytest - ytest.mean()) ** 2).sum())
            a = auc(predicted, labels)
            aucs[name] = a
            tables.append({'capability': cap, 'predictor': name, 'build_rows': 192, 'check_rows': 192,
                           'independent_check_ids': 64, 'heldout_ce_mse': mse,
                           'heldout_ce_r2': None if denominator == 0 else 1 - float(squared_errors.sum()) / denominator,
                           'last_standardized_feature_coefficient': float(beta[-1]),
                           'behavior_incorrect_auc': a, 'incorrect_observations': int(labels.sum()),
                           'correct_observations': int((1 - labels).sum()), 'scope': 'EXPLORATORY_CALIBRATION_NOT_BENCHMARK_EFFECTIVENESS'})
            fits[cap + '/' + name] = {'features': fields, 'build_mean': center.tolist(), 'build_std': scale.tolist(), 'beta': beta.tolist()}
            for row, pred in zip(test, predicted):
                predictions.append({'capability': cap, 'id': row['id'], 'variant': row['variant'], 'predictor': name,
                                    'ce_delta': row['ce_delta'], 'predicted_ce_delta': float(pred),
                                    'behavior_correct': behavior[(row['variant'], row['id'])]})
        ids = sorted({x['id'] for x in test})
        assert len(ids) == 64 and all(sum(x['id'] == key for x in test) == 3 for key in ids)
        indices = rng.integers(0, 64, size=(2000, 64))
        bounds = {}
        for comparison in ['norm_total', 'norm_total_random']:
            differences = residuals[comparison] - residuals['norm_total_functional']
            clustered = np.array([np.mean([d for x, d in zip(test, differences) if x['id'] == key]) for key in ids])
            boots = clustered[indices].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            bounds[comparison] = (float(lo), float(hi))
            comparisons.append({'capability': cap, 'comparison': comparison + '_minus_functional',
                                'heldout_mse_gain': float(differences.mean()), 'ci95_low': float(lo), 'ci95_high': float(hi),
                                'bootstrap_unit': 'example ID with all3 variants retained', 'resamples': 2000,
                                'interval_conditions': 'fixed build-fitted coefficients, source adapter seed and calibration pool; no multiplicity correction'})
        ce_support = all(bounds[name][0] > 0 for name in bounds)
        functional_auc, random_auc = aucs['norm_total_functional'], aucs['norm_total_random']
        behavior_support = bool(labels.sum() >= 5 and (1 - labels).sum() >= 5 and functional_auc is not None and random_auc is not None
                                and functional_auc > .5 and functional_auc >= random_auc)
        gates.append({'capability': cap, 'heldout_ce_additional_information': ce_support,
                      'behavioral_coverage_and_auc_support': behavior_support, 'supports_training_investigation': ce_support and behavior_support})
    behavior_table = []
    for cap in ['math', 'instruction']:
        for split in ['build', 'check']:
            ids = sorted(x['id'] for x in meta if x['capability'] == cap and x['split'] == split)
            base_accuracy = sum(behavior[('base', key)] for key in ids) / 64
            for name in ['base'] + MODELS:
                accuracy = sum(behavior[(name, key)] for key in ids) / 64
                behavior_table.append({'capability': cap, 'split': split, 'variant': name, 'samples': 64,
                                       'accuracy_percent': 100 * accuracy, 'delta_vs_current_base_pp': 100 * (accuracy - base_accuracy),
                                       'scope': 'INDEPENDENT_CALIBRATION_NOT_OFFICIAL_GSM8K_OR_IFEVAL'})
    write_csv(O / 'diagnosis_predictors.csv', tables)
    write_csv(O / 'diagnosis_comparisons.csv', comparisons)
    write_csv(O / 'calibration_behavior.csv', behavior_table)
    write_csv(DATA / 'heldout_predictions.csv', predictions)
    decision = {'stage': 'P1_EXPLORATORY_DIAGNOSIS_COMPLETE', 'analysis_protocol_sha256': sha(protocol_path),
                'capabilities': gates, 'supports_training_investigation': any(x['supports_training_investigation'] for x in gates),
                'base_recheck_errors_without_reselection': behavior_report['base_recheck_errors_without_reselection'],
                'regression_fits_build_only': fits, 'new_training_jobs': 0, 'formal_benchmark_benefit': 'NOT_ESTABLISHED',
                'next_action': 'Review all supporting/counter evidence before launching fixed four-group SW training' if any(x['supports_training_investigation'] for x in gates)
                               else 'Do not launch complex regularized training yet; mechanism/calibration evidence insufficient, not a proof that the method cannot work'}
    (O / 'P1_DECISION.json').write_text(json.dumps(decision, indent=2) + '\n')
    text = ['# 功能敏感表示诊断：探索性独立检验', '',
            '本阶段未进行大模型训练。方向只由build样本构建，线性解释器只在build拟合；check保持独立。此分析受已暴露的实验A启发，非预注册；规则在表示产物生成后、行为汇总和预测结果审查前固定。', '',
            '|能力|预测器|检验CE MSE|检验R²|错误行为AUC|', '|---|---|---:|---:|---:|']
    for x in tables:
        text.append(f"|{x['capability']}|{x['predictor']}|{x['heldout_ce_mse']:.6f}|{str(x['heldout_ce_r2'])[:9]}|{str(x['behavior_incorrect_auc'])[:9]}|")
    text += ['', '## 额外解释力与不确定性', '', '|能力|对照MSE减功能敏感MSE|差值|配对95%区间|', '|---|---|---:|---|']
    for x in comparisons:
        text.append(f"|{x['capability']}|{x['comparison']}|{x['heldout_mse_gain']:+.6f}|[{x['ci95_low']:+.6f},{x['ci95_high']:+.6f}]|")
    text += ['', '正差值表示功能敏感预测器的检验误差较小。区间按64个题目ID聚类，每次保留其三个模型变体；固定build拟合系数，未包含训练种子或拟合样本不确定性，未做多重校正。', '',
             '## 阶段决策', '', '满足预先记录的探索性训练投入条件：' + str(decision['supports_training_investigation']), '',
             '该条件仅决定是否值得投入下一轮训练，不代表新方法有效。校准任务、CE和隐藏状态指标不替代正式阅读、摘要、数学或指令测评。所有负结果均保留。', '',
             '数学来自可见的GSM8K训练题；指令模板范围窄且build/check共享模板。参考回答CE不是完整功能损伤，敏感方向也可能包含有益变化。原始投影量为无符号平方，不能直接区分改善与损伤；标准化系数符号和完整线性拟合保存在CSV/JSON，即使提供额外信息也不能据此宣称该方向必然有害。Base在同一LoRA-enabled引擎中复测，任何差异都不重选样本。正式测试不用于调整方向、rank、层数或系数。', '',
             '完整行为表见calibration_behavior.csv，固定规范见analysis_plan.json；真实大模型参数更新为0。']
    (O / 'mechanism_diagnosis.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({'predictor_rows': len(tables), 'comparison_rows': len(comparisons), 'behavior_rows': len(behavior_table),
                      'supports_training_investigation': decision['supports_training_investigation'], 'capabilities': gates}))


if __name__ == '__main__':
    main()
