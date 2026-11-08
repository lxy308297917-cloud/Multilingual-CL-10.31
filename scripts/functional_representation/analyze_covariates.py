"""Required nuisance robustness check; retain all original exploratory results."""
from pathlib import Path
import csv
import json
import numpy as np
from analyze_diagnosis import R, DATA, BEHAVIOR, O, MODELS, load, sha, auc, write_csv


def main():
    plan_path = R / 'configs/functional_representation_20261109/nuisance_control_plan.json'
    plan = load(plan_path)
    assert plan['always_run_regardless_of_primary_result']
    primary = load(O / 'P1_DECISION.json')
    encoded = {x['id']: x for x in [json.loads(v) for v in (DATA / 'encoded_selected.jsonl').open()]}
    behavior = {}
    for variant in MODELS:
        for x in (json.loads(v) for v in (BEHAVIOR / variant / 'predictions.jsonl').open()):
            behavior[(variant, x['id'])] = bool(x['correct'])
    control = load(DATA / 'manifest.json')['control']
    norm = {'full': 1., 'remove_middle': control['remove_middle_delta_squared_norm'] / control['full_delta_squared_norm'],
            'uniform': control['actual_uniform_delta_squared_norm'] / control['full_delta_squared_norm']}
    rows = []
    for variant in MODELS:
        for x in csv.DictReader((DATA / 'variant_measurements_v1' / variant / 'per_example.csv').open()):
            row = {**x, **{k: float(x[k]) for k in ['base_ce', 'ce_delta', 'answer_total', 'answer_functional', 'answer_random']}}
            row['effective_delta_squared_norm_ratio'] = norm[variant]
            e = encoded[row['id']]
            row['log_prompt_tokens'] = float(np.log1p(e['prefix_length']))
            row['log_answer_tokens'] = float(np.log1p(e['supervised_tokens_including_end_markers']))
            rows.append(row)
    tables, comparisons, fits, gates = [], [], {}, []
    rng = np.random.default_rng(43)
    for cap in ['math', 'instruction']:
        train = sorted([x for x in rows if x['capability'] == cap and x['split'] == 'build'], key=lambda x: (x['id'], x['variant']))
        test = sorted([x for x in rows if x['capability'] == cap and x['split'] == 'check'], key=lambda x: (x['id'], x['variant']))
        assert len(train) == len(test) == 192
        ytrain, ytest = np.array([x['ce_delta'] for x in train]), np.array([x['ce_delta'] for x in test])
        labels = np.array([int(not behavior[(x['variant'], x['id'])]) for x in test])
        errors, aucs = {}, {}
        for name, extra in [('total', []), ('functional', ['answer_functional']), ('random', ['answer_random'])]:
            fields = plan['common_features'] + extra
            a = np.array([[x[k] for k in fields] for x in train], dtype=np.float64)
            b = np.array([[x[k] for k in fields] for x in test], dtype=np.float64)
            center, scale = a.mean(axis=0), a.std(axis=0)
            scale[scale < 1e-12] = 1.
            beta = np.linalg.lstsq(np.column_stack([np.ones(192), (a - center) / scale]), ytrain, rcond=None)[0]
            predicted = np.column_stack([np.ones(192), (b - center) / scale]) @ beta
            errors[name] = (predicted - ytest) ** 2
            aucs[name] = auc(predicted, labels)
            denominator = float(((ytest - ytest.mean()) ** 2).sum())
            tables.append({'capability': cap, 'predictor': name, 'check_rows': 192, 'check_ids': 64,
                           'heldout_ce_mse': float(errors[name].mean()),
                           'heldout_ce_r2': None if denominator == 0 else 1 - float(errors[name].sum()) / denominator,
                           'behavior_incorrect_auc': aucs[name], 'last_standardized_coefficient': float(beta[-1])})
            fits[cap + '/' + name] = {'features': fields, 'mean': center.tolist(), 'std': scale.tolist(), 'beta': beta.tolist()}
        ids = sorted({x['id'] for x in test})
        assert len(ids) == 64 and all(sum(x['id'] == key for x in test) == 3 for key in ids)
        index = rng.integers(0, 64, size=(2000, 64))
        lower_bounds = []
        for name in ['total', 'random']:
            delta = errors[name] - errors['functional']
            clustered = np.array([np.mean([value for row, value in zip(test, delta) if row['id'] == key]) for key in ids])
            assert abs(float(clustered.mean() - delta.mean())) < 1e-10
            lo, hi = np.percentile(clustered[index].mean(axis=1), [2.5, 97.5])
            comparisons.append({'capability': cap, 'comparison': name + '_minus_functional', 'heldout_mse_gain': float(delta.mean()),
                                'ci95_low': float(lo), 'ci95_high': float(hi), 'bootstrap_unit': 'example ID retaining all3 variants'})
            lower_bounds.append(float(lo))
        behavior_support = bool(labels.sum() >= 5 and (1 - labels).sum() >= 5 and aucs['functional'] is not None and aucs['random'] is not None
                                and aucs['functional'] > .5 and aucs['functional'] >= aucs['random'])
        original = next(x['supports_training_investigation'] for x in primary['capabilities'] if x['capability'] == cap)
        robust = all(x > 0 for x in lower_bounds) and behavior_support
        gates.append({'capability': cap, 'original_primary_gate': original, 'nuisance_controlled_gate': robust,
                      'combined_support_for_training_investigation': original and robust})
    write_csv(O / 'nuisance_predictors.csv', tables)
    write_csv(O / 'nuisance_comparisons.csv', comparisons)
    decision = {'scope': 'Combined exploratory primary+nuisance evidence; not method effectiveness',
                'supports_training_investigation': any(x['combined_support_for_training_investigation'] for x in gates),
                'capabilities': gates, 'primary_decision_sha256': sha(O / 'P1_DECISION.json'),
                'nuisance_plan_sha256': sha(plan_path), 'build_only_fits': fits,
                'new_training_jobs': 0, 'formal_benchmark_benefit': 'NOT_ESTABLISHED'}
    (O / 'P1_COMBINED_DECISION.json').write_text(json.dumps(decision, indent=2) + '\n')
    text = ['# 功能敏感表示：P1综合诊断', '', '满足进一步训练投入条件：**' + str(decision['supports_training_investigation']) + '**。此项只判断是否值得继续验证，不代表方法有效。', '',
            '保留最初的总漂移/参数范数/随机方向比较，并额外共同控制适配前CE、提示长度和监督长度，检查额外信息是否只是样本难度或长度。补充规则在首次统计执行和行为汇总审查前固定，全部分析为探索性，非预注册。', '',
            '|能力|混杂控制后的预测器|检验CE MSE|错误行为AUC|', '|---|---|---:|---:|']
    for x in tables:
        text.append(f"|{x['capability']}|{x['predictor']}|{x['heldout_ce_mse']:.6f}|{str(x['behavior_incorrect_auc'])[:9]}|")
    text += ['', '## 正负证据与决定', '']
    for x in gates:
        text.append(f"- {x['capability']}：原始探索门槛={x['original_primary_gate']}；混杂控制门槛={x['nuisance_controlled_gate']}；合并支持={x['combined_support_for_training_investigation']}。")
    text += ['', '原始完整结果见mechanism_diagnosis.md、diagnosis_predictors.csv、diagnosis_comparisons.csv；新增完整结果见nuisance_predictors.csv、nuisance_comparisons.csv。校准实际正确率见calibration_behavior.csv。', '',
             '按64个题目ID配对bootstrap，保留同题三个模型，条件于固定拟合和单训练seed，未做多重校正。方向构造、线性拟合都只用build；check不参与构建或训练。', '',
             '数学训练题可能在预训练中出现；指令只有8类共享模板，Base正确筛选改变了样本分布。平方漂移不区分有益和有害变化，CE与行为不是同一个指标。不能把本次代理解释力直接等同于正式benchmark收益，也不能由阴性结果断言此方法永远无效。', '',
             '若合并条件不满足，暂不启动复杂表示约束训练。若满足，仍需审查全部反例和成本，完成固定预算四组训练和正式端点评测，才能判断方法价值。动态预算和顺序CL仍未启动。']
    (O / 'diagnosis.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({'nuisance_predictor_rows': len(tables), 'comparison_rows': len(comparisons), 'combined_training_support': decision['supports_training_investigation'], 'capabilities': gates}))


if __name__ == '__main__':
    main()
