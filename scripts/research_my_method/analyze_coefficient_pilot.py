"""Audit every saved MCQ outcome and report all predeclared paired ACC effects."""
import argparse, collections, hashlib, json, math, sys
from pathlib import Path
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_rows(path, protocol, condition_key):
    rows = [json.loads(x) for x in Path(path).open()]
    expected = {(sid, lang) for sid in protocol['confirmation_ids'] for lang in protocol['languages']}
    groups = collections.defaultdict(dict)
    for row in rows:
        condition = row[condition_key]; key = (row['semantic_id'], row['language'])
        assert key in expected and key not in groups[condition]
        gold = protocol['encoded'][f'{key[0]}|{key[1]}|0']['gold']
        assert row['gold'] == 'ABCD'[gold]
        values = row['option_scores']; assert len(values) == 4 and all(math.isfinite(x) for x in values)
        pred = max(range(4), key=lambda i: values[i])
        assert row['predicted'] == 'ABCD'[pred]
        assert row['correct'] == (pred == gold)
        groups[condition][key] = row
    assert groups and all(set(values) == expected for values in groups.values())
    return groups


def main():
    p = argparse.ArgumentParser()
    for key in ['project', 'run', 'protocol', 'preregistration', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--dev-runs', type=Path)
    a = p.parse_args(); assert not a.output.exists(); a.output.mkdir(parents=True)
    sys.path.insert(0, str(a.project/'scripts/bench_v2'))
    from paired_statistics import paired_macro
    protocol = json.loads(a.protocol.read_text())
    registration = json.loads(a.preregistration.read_text())
    assert registration['protocol_sha256'] == sha(a.protocol)
    done = json.loads((a.run/'DONE.json').read_text())
    assert done['status'] == 'EXPLORATORY_COMPLETE' and done['protocol_sha256'] == sha(a.protocol)
    groups = audit_rows(a.run/'predictions.jsonl', protocol, 'condition')
    expected = {'W1'} | {f'{condition}-fraction{fraction}' for condition in protocol['conditions'] for fraction in protocol['norm_fractions']}
    assert set(groups) == expected
    summaries = {}
    vectors = {}
    for condition, values in groups.items():
        summaries[condition] = {}; vectors[condition] = {}
        for lang in protocol['languages']:
            vector = np.array([values[sid, lang]['correct'] for sid in protocol['confirmation_ids']], dtype=float)
            vectors[condition][lang] = vector
            correct, n = int(vector.sum()), len(vector)
            actual = {'n': n, 'correct': correct, 'acc': correct/n}
            assert done['results'][condition][lang]['n'] == n
            assert done['results'][condition][lang]['correct'] == correct
            assert abs(done['results'][condition][lang]['acc']-correct/n) < 1e-12
            summaries[condition][lang] = actual
    interventions = [json.loads(x) for x in (a.run/'interventions.jsonl').open()]
    assert len(interventions) == 14 and {x['condition'] for x in interventions} == expected-{'W1'}
    for row in interventions:
        assert row['all_other_weights_frozen'] and row['max_per_matrix_relative_norm_error'] < 1e-4
        target = row['target_effective_weight_norm']
        assert abs(row['measured_effective_weight_norm']-target)/target < 1e-4
    contrasts = []
    def contrast(treatment, control, primary=False):
        for lang in protocol['languages']:
            stats = paired_macro(100*vectors[treatment][lang], 100*vectors[control][lang],
                                 repeats=10000, seed=20261103)
            stats['unit'] = 'paired semantic IDs; identical resampling across languages'
            contrasts.append({'treatment': treatment, 'control': control, 'language': lang,
                              'primary_exploratory': primary and lang in ['sw', 'ha'],
                              'scale': 'percentage points', **stats})
    for fraction in protocol['norm_fractions']:
        contrast(f'block_shared-fraction{fraction}', f'block_shuffled-fraction{fraction}', fraction == .1)
        contrast(f'block_shared-fraction{fraction}', f'block_random-fraction{fraction}')
        contrast(f'reuse_shared-fraction{fraction}', f'reuse_shuffled-fraction{fraction}')
        contrast(f'reuse_shared-fraction{fraction}', f'reuse_random-fraction{fraction}')
    for condition in sorted(expected-{'W1'}): contrast(condition, 'W1')
    dev = {}; precision = []
    if a.dev_runs:
        for folder in sorted(a.dev_runs.iterdir()):
            if not (folder/'DONE.json').exists(): continue
            saved = json.loads((folder/'DONE.json').read_text())
            assert saved['status'] == 'EXPLORATORY_DEV_COMPLETE'
            assert saved['predictions_sha256'] == sha(folder/'predictions.jsonl')
            assert saved['protocol_sha256'] == sha(folder/'protocol.json')
            assert json.loads((folder/'protocol.json').read_text())['cohort_protocol_sha256'] == sha(a.protocol)
            records = audit_rows(folder/'predictions.jsonl', protocol, 'label')
            assert len(records) == 1 and saved['label'] in records
            key = saved['label']; values = records[key]
            dev[key] = {lang: {'n': len(protocol['confirmation_ids']),
                'correct': sum(values[sid, lang]['correct'] for sid in protocol['confirmation_ids']),
                'acc': np.mean([values[sid, lang]['correct'] for sid in protocol['confirmation_ids']]).item()}
                for lang in protocol['languages']}
            if key == 'SW-SST-NoRefresh-BF16Export':
                for lang in protocol['languages']:
                    rounded = np.array([values[sid, lang]['correct'] for sid in protocol['confirmation_ids']], dtype=float)
                    result = paired_macro(100*rounded, 100*vectors['W1'][lang], repeats=10000, seed=20261103)
                    result['unit'] = 'paired semantic IDs'
                    precision.append({'language': lang, 'comparison': 'rounded BF16 weights minus native FP32 masters; both FP32/eager',
                                      'scale': 'percentage points', **result})
    report = {'status': 'VERIFIED_EXPLORATORY_COMPLETE', 'source_predictions_sha256': sha(a.run/'predictions.jsonl'),
        'protocol_sha256': sha(a.protocol), 'preregistration_sha256': sha(a.preregistration),
        'statistics_code_sha256': sha(a.project/'scripts/bench_v2/paired_statistics.py'),
        'semantic_ids': len(protocol['confirmation_ids']), 'languages': protocol['languages'],
        'conditions': len(groups), 'predictions_verified': sum(map(len, groups.values())),
        'summaries': summaries, 'contrasts': contrasts, 'sequential_dev': dev, 'precision_control': precision,
        'coverage': json.loads((a.run/'coverage.json').read_text()),
        'formal_test_used': False, 'adoption_claim_allowed': False,
        'limitations': ['EN/SW dev historically exposed; not pristine confirmation.',
                       'Same-item difficulty and translation covariance not fully isolated.',
                       'One fixed shuffled control; intervals are unadjusted exploratory item bootstrap.',
                       'One finite FP32-master perturbation is not full sequential training or BF16 deployment.']}
    (a.output/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# 有限配对功能干预：探索性结果', '',
        '逐题gold、argmax、ACC及所有范数匹配已独立复核。以下不是正式benchmark或完整方法训练。', '',
        '| 条件 | EN ACC | SW ACC | HA ACC |', '|---|---:|---:|---:|']
    for condition in ['W1'] + [f'{c}-fraction{f}' for f in protocol['norm_fractions'] for c in protocol['conditions']]:
        lines.append('| '+condition+' | '+' | '.join('%.2f%%'%(100*summaries[condition][lang]['acc']) for lang in protocol['languages'])+' |')
    lines += ['', '## 预登记主要探索对比', '', '| 10%：配对移除−打乱移除 | ACC差（百分点） | 95%配对区间 |', '|---|---:|---:|']
    for row in contrasts:
        if row['primary_exploratory']:
            lines.append('| '+row['language']+' | %+.2f | [%+.2f, %+.2f] |'%(row['delta'],row['ci95_low'],row['ci95_high']))
    lines += ['', '同时报告SW保持与HA收益，不计算事后加权综合分。EN、30%强度及其他方向属于次要探索；完整对比全部保存在analysis.json。', '',
              '边界：历史dev暴露、题目难度/翻译质量混杂、单次打乱、有限FP32干预。不能据本表宣称已找到有效创新方法。']
    if dev:
        lines += ['', '## 固定阶段的同协议开发ACC', '',
                  '均为实际导出权重提升至FP32/eager，0-shot，与历史正式BF16/5-shot分数分开。', '',
                  '| 阶段 | EN | SW | HA |', '|---|---:|---:|---:|']
        for name in ['Base', 'SW-FFT-Final', 'SW-HA-FFT-Final']:
            if name in dev:
                lines.append('| '+name+' | '+' | '.join('%.2f%%'%(100*dev[name][lang]['acc']) for lang in protocol['languages'])+' |')
    if precision:
        lines += ['', '## 舍入权重控制', '',
                  '以下为BF16权重精确提升至FP32后的分数减去native FP32 master分数；计算方式相同。', '',
                  '| 语言 | ACC差（百分点） | 95%配对区间 |', '|---|---:|---:|']
        for row in precision:
            lines.append('| '+row['language']+' | %+.2f | [%+.2f, %+.2f] |'%(row['delta'],row['ci95_low'],row['ci95_high']))
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k in ['status', 'conditions', 'predictions_verified', 'semantic_ids']}))


if __name__ == '__main__': main()
