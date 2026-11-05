"""All predeclared roles, paired task deltas and frozen route gate; no best-score omission."""
from profile_paths import resolve
import json, collections, csv, math, argparse, time, re
from pathlib import Path
from common import *
import numpy as np
R = ROOT / 'results/math_sft_base_fft'
DOC = ROOT / 'research/math_sft_base_fft'

def paired(a, b):
    assert len(a) == len(b) and a
    n = len(a)
    wrong = sum((x and (not y) for x, y in zip(a, b)))
    right = sum((not x and y for x, y in zip(a, b)))
    rng = np.random.default_rng(42)
    draw = rng.multinomial(n, [wrong / n, (n - right - wrong) / n, right / n], size=10000)
    delta = (draw[:, 2] - draw[:, 0]) / n * 100
    ci = np.percentile(delta, [2.5, 97.5]).tolist()
    return {'n': n, 'before_correct': sum(a), 'after_correct': sum(b), 'before_acc': sum(a) / n, 'after_acc': sum(b) / n, 'delta_pp': (sum(b) - sum(a)) / n * 100, 'right_to_wrong': wrong, 'wrong_to_right': right, 'paired_bootstrap95_pp': ci, 'bootstrap_replicates': 10000, 'bootstrap_seed': 42, 'inference_scope': 'paired question uncertainty, single training seed; not training-seed stability or multiplicity-adjusted confirmatory inference'}

def math_gate_passes(cs):
    assert len(cs) == 2 and {r['task'] for r in cs} == {'mgsm_rev2', 'msvamp'}
    return all((r['delta_pp'] >= 2 - 1e-10 for r in cs)) and any((r['paired_bootstrap95_pp'][0] > 0 for r in cs)) and all((r['paired_bootstrap95_pp'][1] >= 0 for r in cs))

def scalar_general(folder):
    scores = json.loads((folder / 'scores.json').read_text())
    core = scores['core']
    mmlu = scores['mmlu']['mmlu_locked']
    code = json.loads((folder / 'code-execution/scores.json').read_text())
    assert code['status'] == 'COMPLETE' and code['n'] == 164
    fs = list((folder / 'core/results').glob('**/samples_sft_v1_gsm8k_*.jsonl'))
    assert len(fs) == 1
    filtered = [x['filtered_resps'][0] for x in rows(fs[0]) if x['filter'] == 'flexible-extract']
    assert len(filtered) == 1319
    parsed = sum((v is not None and str(v).strip() not in ['', '[invalid]'] for v in filtered))
    return {'GSM8K_flexible_parse_rate': parsed / 1319, 'GSM8K_flexible_unparsed_count': 1319 - parsed, 'GSM8K_flexible': core['sft_v1_gsm8k']['exact_match,flexible-extract'], 'GSM8K_strict': core['sft_v1_gsm8k']['exact_match,strict-match'], 'MMLU': mmlu['acc,none'], 'IFEval_prompt_strict': core['sft_v1_ifeval']['prompt_level_strict_acc,none'], 'IFEval_instruction_strict': core['sft_v1_ifeval']['inst_level_strict_acc,none'], 'IFEval_prompt_loose': core['sft_v1_ifeval']['prompt_level_loose_acc,none'], 'IFEval_instruction_loose': core['sft_v1_ifeval']['inst_level_loose_acc,none'], 'HumanEval_plus': code['pass_at_1_plus'], 'HumanEval_base': code['pass_at_1_base']}

def general_flags(folder):
    vals = {k: {} for k in ['GSM8K_flexible', 'MMLU', 'IFEval_prompt_strict', 'HumanEval_plus']}
    for f in (folder / 'mmlu/results').glob('**/samples_*.jsonl'):
        task = re.sub('_[0-9]{4}-[0-9]{2}-[0-9]{2}T.*$', '', f.stem)
        for r in rows(f):
            vals['MMLU'][task + ':' + str(r['doc_id'])] = {'correct': bool(r['acc']), 'identity': [r['doc_hash'], r['prompt_hash'], r['target_hash']]}
    for task, metric, filter_name, key in [('sft_v1_gsm8k', 'exact_match', 'flexible-extract', 'GSM8K_flexible'), ('sft_v1_ifeval', 'prompt_level_strict_acc', 'none', 'IFEval_prompt_strict')]:
        fs = list((folder / 'core/results').glob('**/samples_' + task + '_*.jsonl'))
        assert len(fs) == 1
        for r in rows(fs[0]):
            if r['filter'] == filter_name:
                vals[key][str(r['doc_id'])] = {'correct': bool(r[metric]), 'identity': [r['doc_hash'], r['prompt_hash'], r['target_hash']]}
    generated = {r['id']: r['prompt_token_sha256'] for r in rows(folder / 'code/generation/predictions.jsonl')}
    for r in json.loads((folder / 'code-execution/model_evaluation-results.json').read_text()):
        vals['HumanEval_plus'][r['task_id']] = {'correct': bool(r['plus_pass']), 'identity': [generated[r['task_id']]]}
    assert {k: len(v) for k, v in vals.items()} == {'GSM8K_flexible': 1319, 'MMLU': 14042, 'IFEval_prompt_strict': 541, 'HumanEval_plus': 164}
    return vals

def write_expanded_tables(roles, comparisons, dw_comparisons, general_comparisons):
    metrics = ['mgsm_rev2_acc', 'msvamp_acc', 'polymath_DW_ACC_percent', 'GSM8K_flexible', 'MMLU', 'IFEval_prompt_strict', 'HumanEval_plus']
    labels = ['MGSM-Rev2', 'MSVAMP', 'PolyMath DW-ACC', 'EN GSM8K flexible', 'EN MMLU', 'IFEval prompt strict', 'HumanEval+']

    def value(r, key):
        return r[key] * (1 if key == 'polymath_DW_ACC_percent' else 100)
    content = '# 三语言完整节点与预算差值\n\n所有成绩按百分数报告，差值为百分点。三个语言分别从同一Base独立训练。\n\n'
    for lang in ['sw', 'bn', 'ja']:
        rs = {r['epoch']: r for r in roles if r['language'] == lang}
        content += '## ' + lang.upper() + '\n\n|模型／对比|' + '|'.join(labels) + '|\n|---|' + '|'.join(['---:'] * 7) + '|\n'
        for ep in [0, 1, 2]:
            content += '|' + rs[ep]['role'] + '|' + '|'.join((f'{value(rs[ep], k):.2f}' for k in metrics)) + '|\n'
        for before, after in [(0, 1), (0, 2), (1, 2)]:
            content += f'|Δ E{after}−' + ('Base' if before == 0 else 'E' + str(before)) + '|' + '|'.join((f'{value(rs[after], k) - value(rs[before], k):+.2f}' for k in metrics)) + '|\n'
        content += '\n'
    content += '各节点与差值完整保留；没有逐语言挑选最佳epoch。更小预算上的高分不构成对第二轮训练的事后早停许可，也不能从一次训练轨迹直接确定最优epoch。配对区间见[第二轮相对第一轮](epoch2_minus_epoch1.md)及机器可读全量配对比较。\n'
    (DOC / 'mathoctopus_epoch_comparison.md').write_text(content)
    counts = '\n## 数学评分与生成诊断（完整分母）\n\n|语言|节点|任务|正确／总数|ACC%|数字可解析率%|生成截断率%|strict格式ACC%|\n|---|---|---|---:|---:|---:|---:|---:|\n'
    for r in roles:
        for task in ['mgsm_rev2', 'msvamp']:
            counts += f"|{r['language']}|{r['role']}|{task}|{r[task + '_correct']}/{r[task + '_n']}|{r[task + '_acc'] * 100:.2f}|{r[task + '_parse_rate'] * 100:.2f}|{r[task + '_cap_rate'] * 100:.2f}|{r[task + '_strict_acc'] * 100:.2f}|\n"
    counts += '\n## PolyMath各难度（官方表达式等价）\n\n|语言|节点|难度|正确／总数|ACC%|boxed可解析率%|生成截断率%|\n|---|---|---|---:|---:|---:|---:|\n'
    for r in roles:
        for level in ['low', 'medium', 'high', 'top']:
            counts += f"|{r['language']}|{r['role']}|{level}|{r['polymath_' + level + '_correct']}/125|{r['polymath_' + level + '_acc'] * 100:.2f}|{r['polymath_' + level + '_box_parse_rate'] * 100:.2f}|{r['polymath_' + level + '_cap_rate'] * 100:.2f}|\n"
    counts += '\n## MGSM/MSVAMP逐题配对差异\n\n|语言|对比|任务|变化（百分点）|正确变错|错误变对|配对95%区间（百分点）|\n|---|---|---|---:|---:|---:|---|\n'
    for x in comparisons:
        if x['task'] not in ['mgsm_rev2', 'msvamp']:
            continue
        lo, hi = x['paired_bootstrap95_pp']
        counts += f"|{x['language']}|E{x['after_epoch']}−" + ('Base' if x['before_epoch'] == 0 else 'E' + str(x['before_epoch'])) + f"|{x['task']}|{x['delta_pp']:+.2f}|{x['right_to_wrong']}|{x['wrong_to_right']}|[{lo:+.2f}, {hi:+.2f}]|\n"
    counts += '\n所有不可解析、错误和截断样本保留在完整分母。strict仅是格式辅助口径；高截断率与低分相关不证明加长生成必然答对。每节点平均输出token及无答案逐题标记见机器表；输出语言另外作有限自动识别诊断。\n'
    p = DOC / 'target_math_results.md'
    p.write_text(p.read_text() + counts)
    table = '# 第二轮相对第一轮：全部主指标与配对区间\n\n|语言|任务|E2−E1（百分点）|配对95%区间（百分点）|正确变错|错误变对|\n|---|---|---:|---|---:|---:|\n'
    for lang in ['sw', 'bn', 'ja']:
        for key, label in zip(metrics, labels):
            if key == 'polymath_DW_ACC_percent':
                x = next((x for x in dw_comparisons if x['language'] == lang and x['before_epoch'] == 1 and (x['after_epoch'] == 2)))
                lo, hi = x['paired_stratified_bootstrap95_pp']
                table += f"|{lang}|{label}（未舍入计数）|{x['delta_pp_unrounded']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|分难度另列|分难度另列|\n"
            else:
                source = comparisons if key in ['mgsm_rev2_acc', 'msvamp_acc'] else general_comparisons
                task = {'mgsm_rev2_acc': 'mgsm_rev2', 'msvamp_acc': 'msvamp'}.get(key, key)
                x = next((x for x in source if x['language'] == lang and x['task'] == task and (x['before_epoch'] == 1) and (x['after_epoch'] == 2)))
                lo, hi = x['paired_bootstrap95_pp']
                table += f"|{lang}|{label}|{x['delta_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|{x['right_to_wrong']}|{x['wrong_to_right']}|\n"
    table += '\nPolyMath区间按难度分层重采样、以官方1/2/4/8权重作用于未舍入正确率；官方显示DW-ACC先舍入再加权，两者差值可能有小的舍入差异。所有区间为固定seed42模型的逐题配对10000次bootstrap，不代表跨训练种子稳定性，也未做多重比较校正。E1−Base和E2−Base的同口径统计保留在对应CSV/JSON中。\n'
    (DOC / 'epoch2_minus_epoch1.md').write_text(table)

def main():
    assert (R / 'cpu-drain-routeA-v1/DONE.json').exists()
    selection = json.loads((R / 'base-language-screening-v1/selection.json').read_text())
    langs = selection['selected_languages']
    out = R / 'routeA-analysis-v1'
    assert not out.exists()
    out.mkdir()
    base = []
    for i in [1, 2, 3]:
        base += rows(R / f'base-census-worker{i}-v1/predictions.jsonl')
    general_base = scalar_general(R / 'base-general-v1')
    base_flags = general_flags(R / 'base-general-v1')
    roles = []
    comparisons = []
    dw_comparisons = []
    general_comparisons = []
    gates = []
    source = {}
    for lang in langs:
        raw = {0: [r for r in base if r['lang'] == lang]}
        poly = {0: rows(R / 'base-census-polymath-v1' / f'{lang}-score/scored.jsonl')}
        general = {0: general_base}
        flags = {0: base_flags}
        train = f'mathoctopus_{lang}_full_2epochs_seed42_v1'
        cfg = json.loads((ROOT / 'configs/math_sft_base_fft' / (train + '.budget.json')).read_text())
        training = json.loads((Path(resolve('CHECKPOINTS')) / train / 'TRAINING_DONE.json').read_text())
        for ep in [1, 2]:
            folder = R / 'A' / lang / f'epoch{ep}'
            assert (folder / 'GPU_PHASE_DONE.json').exists()
            raw[ep] = rows(folder / 'math/predictions.jsonl')
            poly[ep] = rows(folder / 'polymath/scored.jsonl')
            general[ep] = scalar_general(folder / 'general')
            flags[ep] = general_flags(folder / 'general')
            source[lang + ':E' + str(ep)] = {str(f.relative_to(folder)): sha(f) for f in [folder / 'math/predictions.jsonl', folder / 'polymath/scored.jsonl', folder / 'general/scores.json', folder / 'general/code-execution/scores.json']}
        for ep in [0, 1, 2]:
            record = {'language': lang, 'epoch': ep, 'role': 'Base' if ep == 0 else 'FFT-E' + str(ep), 'optimizer_steps': 0 if ep == 0 else cfg['epoch' + str(ep) + '_step'], 'input_tokens': 0 if ep == 0 else training['epochs'][ep - 1]['cumulative_input_tokens'], 'shifted_supervised_tokens': 0 if ep == 0 else training['epochs'][ep - 1]['cumulative_shifted_supervised_tokens'], 'source_records': cfg['samples_per_unpadded_epoch'], 'actual_exposure_samples': 0 if ep == 0 else training['epochs'][ep - 1]['cumulative_samples']}
            for task, n in [('mgsm_rev2', 250), ('msvamp', 1000)]:
                xs = [r for r in raw[ep] if r['benchmark'] == task]
                assert len(xs) == n
                record[task + '_correct'] = sum((r['correct'] for r in xs))
                record[task + '_n'] = n
                record[task + '_acc'] = sum((r['correct'] for r in xs)) / n
                record[task + '_parse_rate'] = sum((r['parsed'] is not None for r in xs)) / n
                record[task + '_cap_rate'] = sum((r['finish_reason'] == 'length' for r in xs)) / n
                record[task + '_strict_acc'] = sum((r['strict_correct'] for r in xs)) / n
                record[task + '_mean_output_tokens'] = sum((r['generated_tokens'] for r in xs)) / n
            vals = []
            for level in ['low', 'medium', 'high', 'top']:
                xs = [r for r in poly[ep] if r['level'] == level]
                assert len(xs) == 125
                v = sum((r['correct'] for r in xs)) / 125
                record['polymath_' + level + '_acc'] = v
                record['polymath_' + level + '_correct'] = sum((r['correct'] for r in xs))
                record['polymath_' + level + '_box_parse_rate'] = sum((r['box_parse'] for r in xs)) / 125
                generated = [r for r in raw[ep] if r['benchmark'] == 'polymath' and r['level'] == level]
                assert len(generated) == 125
                record['polymath_' + level + '_cap_rate'] = sum((r['finish_reason'] == 'length' for r in generated)) / 125
                record['polymath_' + level + '_mean_output_tokens'] = sum((r['generated_tokens'] for r in generated)) / 125
                vals.append(v)
            record['polymath_DW_ACC_percent'] = round(sum((2 ** i * round(v * 100, 1) for i, v in enumerate(vals))) / 15, 1)
            record.update(general[ep])
            roles.append(record)
        for before, after in [(0, 1), (0, 2), (1, 2)]:
            for task in ['mgsm_rev2', 'msvamp']:
                aa = {r['id']: r for r in raw[before] if r['benchmark'] == task}
                bb = {r['id']: r for r in raw[after] if r['benchmark'] == task}
                assert set(aa) == set(bb)
                ids = sorted(aa)
                assert all((aa[k]['prompt_token_sha256'] == bb[k]['prompt_token_sha256'] for k in ids))
                comparisons.append(dict(language=lang, before_epoch=before, after_epoch=after, task=task, **paired([aa[k]['correct'] for k in ids], [bb[k]['correct'] for k in ids])))
            for level in ['low', 'medium', 'high', 'top']:
                aa = {r['id']: r for r in poly[before] if r['level'] == level}
                bb = {r['id']: r for r in poly[after] if r['level'] == level}
                assert set(aa) == set(bb)
                ids = sorted(aa)
                pa = {r['id']: r['prompt_token_sha256'] for r in raw[before] if r['benchmark'] == 'polymath' and r['level'] == level}
                pb = {r['id']: r['prompt_token_sha256'] for r in raw[after] if r['benchmark'] == 'polymath' and r['level'] == level}
                assert pa == pb and set(pa) == set(aa), (lang, level, 'PolyMath source or prompt identity changed')
                comparisons.append(dict(language=lang, before_epoch=before, after_epoch=after, task='polymath_' + level, **paired([aa[k]['correct'] for k in ids], [bb[k]['correct'] for k in ids])))
        for before, after in [(0, 1), (0, 2), (1, 2)]:
            rng = np.random.default_rng(42)
            draws = np.zeros(10000)
            delta = 0.0
            details = []
            for i, level in enumerate(['low', 'medium', 'high', 'top']):
                aa = {r['id']: r for r in poly[before] if r['level'] == level}
                bb = {r['id']: r for r in poly[after] if r['level'] == level}
                ids = sorted(aa)
                assert set(ids) == set(bb)
                neg = sum((aa[k]['correct'] and (not bb[k]['correct']) for k in ids))
                pos = sum((not aa[k]['correct'] and bb[k]['correct'] for k in ids))
                n = len(ids)
                sample = rng.multinomial(n, [neg / n, (n - pos - neg) / n, pos / n], size=10000)
                weight = 2 ** i / 15
                draws += weight * (sample[:, 2] - sample[:, 0]) / n * 100
                delta += weight * (pos - neg) / n * 100
                details.append({'level': level, 'n': n, 'right_to_wrong': neg, 'wrong_to_right': pos})
            dw_comparisons.append({'language': lang, 'before_epoch': before, 'after_epoch': after, 'metric': 'PolyMath DW-ACC (unrounded counts, official weights)', 'delta_pp_unrounded': delta, 'paired_stratified_bootstrap95_pp': np.percentile(draws, [2.5, 97.5]).tolist(), 'bootstrap_replicates': 10000, 'seed': 42, 'difficulty_pair_counts': details, 'note': 'Official displayed DW-ACC first rounds per-level percentages; inferential statistic uses exact paired rates, independent resampling within each difficulty.'})
        for before, after in [(0, 1), (0, 2), (1, 2)]:
            for task in ['GSM8K_flexible', 'MMLU', 'IFEval_prompt_strict', 'HumanEval_plus']:
                aa = flags[before][task]
                bb = flags[after][task]
                assert set(aa) == set(bb)
                ids = sorted(aa)
                assert all((aa[k]['identity'] == bb[k]['identity'] for k in ids)), (lang, task, 'source or prompt identity changed')
                general_comparisons.append(dict(language=lang, before_epoch=before, after_epoch=after, task=task, **paired([aa[k]['correct'] for k in ids], [bb[k]['correct'] for k in ids])))
        for ep in [1, 2]:
            cs = [r for r in comparisons if r['language'] == lang and r['before_epoch'] == 0 and (r['after_epoch'] == ep) and (r['task'] in ['mgsm_rev2', 'msvamp'])]
            assert len(cs) == 2
            passing = math_gate_passes(cs)
            gates.append({'language': lang, 'epoch': ep, 'passes': passing, 'metrics': cs})
    epoch_counts = {str(ep): sum((x['passes'] for x in gates if x['epoch'] == ep)) for ep in [1, 2]}
    success = any((v >= 2 for v in epoch_counts.values()))
    decision = {'route': 'A', 'status': 'PASS' if success else 'FAIL_FIXED_E1_AND_E2_GATE', 'passing_languages_per_same_epoch': epoch_counts, 'gate_details': gates, 'route_B': 'NOT_RUN_CONDITION_NOT_MET' if success else 'AUTHORIZED_CONDITIONAL_READY_FOR_RESOURCE_AND_DATA_AUDIT', 'EN_decline_used_as_failure': False, 'all_epochs_and_languages_retained': True, 'scope': 'BENCHMARK_GUIDED_EXPLORATORY_SINGLE_SEED'}
    gate_file = R / 'routeA-math-gate-v1/decision.json'
    if gate_file.exists():
        prior = json.loads(gate_file.read_text())
        assert prior['status'] == decision['status'] and prior['passing_languages_per_same_epoch'] == epoch_counts and (prior['gate_details'] == gates), 'Interim and complete frozen gate disagree'
        decision['earlier_math_gate_sha256'] = sha(gate_file)
    dump(out / 'all_roles.json', roles)
    dump(out / 'paired_comparisons.json', comparisons)
    dump(out / 'polymath_dw_paired_comparisons.json', dw_comparisons)
    dump(out / 'route_decision.json', decision)
    dump(out / 'source_identity.json', source)
    dump(out / 'general_paired_comparisons.json', general_comparisons)
    with (out / 'all_roles.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(roles[0]))
        w.writeheader()
        w.writerows(roles)
    with (out / 'general_paired_comparisons.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(general_comparisons[0]))
        w.writeheader()
        w.writerows(general_comparisons)
    with (out / 'paired_comparisons.csv').open('w', newline='') as f:
        fields = list(comparisons[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(comparisons)
    table = '|语言|节点|步数|MGSM ACC%|MSVAMP ACC%|PolyMath DW-ACC%|\n|---|---|---:|---:|---:|---:|\n' + ''.join((f"|{r['language']}|{r['role']}|{r['optimizer_steps']}|{r['mgsm_rev2_acc'] * 100:.2f}|{r['msvamp_acc'] * 100:.2f}|{r['polymath_DW_ACC_percent']:.1f}|\n" for r in roles))
    txt = '# 路线A：完整两轮的实际目标数学结果\n\n' + table + '\n每语言是独立Base初始化，非顺序语言训练。两个epoch均完整报告，未因英语退化早停。PolyMath是官方分难度表达式等价，DW-ACC依照官方舍入与权重。完整分难度、正确数、解析率、截断和逐题配对bootstrap95%区间见CSV/JSON。测试已用于选语言和路线，属于探索性结果；仅seed42，不代表跨种子稳定性。\n'
    (DOC / 'target_math_results.md').write_text(txt)
    write_expanded_tables(roles, comparisons, dw_comparisons, general_comparisons)
    metrics = ['GSM8K_flexible', 'MMLU', 'IFEval_prompt_strict', 'HumanEval_plus']
    gt = '|语言|节点|EN GSM8K flexible%|EN MMLU%|IFEval prompt strict%|HumanEval+ pass@1%|\n|---|---|---:|---:|---:|---:|\n' + ''.join(('|' + r['language'] + '|' + r['role'] + '|' + '|'.join((f'{r[k] * 100:.2f}' for k in metrics)) + '|\n' for r in roles))
    aux_keys = ['GSM8K_strict', 'GSM8K_flexible_parse_rate', 'IFEval_instruction_strict', 'IFEval_prompt_loose', 'IFEval_instruction_loose', 'HumanEval_base']
    aux = '|语言|节点|GSM8K strict%|GSM8K flexible parse%|IFEval instruction strict%|IFEval prompt loose%|IFEval instruction loose%|HumanEval base%|\n|---|---|---:|---:|---:|---:|---:|---:|\n' + ''.join(('|' + r['language'] + '|' + r['role'] + '|' + '|'.join((f'{r[k] * 100:.2f}' for k in aux_keys)) + '|\n' for r in roles))
    (DOC / 'general_retention_results.md').write_text('# 通用能力完整成绩\n\n' + gt + '\n## 辅助评分口径（全部节点）\n\n' + aux + '\nBase为同一共享参照。GSM8K strict、IFEval instruction strict及loose、HumanEval原始子集另列机器可读表，不能替换各自主要指标。HumanEval+经过实际隔离执行，非静态估算。MMLU固定5-shot候选对数似然。四项主指标的E1-Base/E2-Base/E2-E1均另有逐题配对变化、正确变错/错误变对及95%区间，且核对相同题目/提示身份；不把细小负数自动解释为稳健遗忘。\n')
    (DOC / 'route_selection_decision.md').write_text('# 冻结门槛的路线决定\n\n状态：' + decision['status'] + '。同一epoch通过语言数：' + str(epoch_counts) + '。\n\n每语言MGSM与MSVAMP各至少+2pp，至少一项配对95%下界>0；同一epoch>=2/3语言即A通过。全部九语言Base、三语言两节点和所有通用结果保留。英语下降不构成失败。路线B：' + decision['route_B'] + '。\n\n|路线|训练／评测状态|冻结门槛与决定|\n|---|---|---|\n|A：全部有效MathOctopus×2轮|COMPLETE|' + decision['status'] + '；同一epoch通过数：' + str(epoch_counts) + '|\n|B：mCoT-MATH 30k×2轮|NOT_RUN|' + decision['route_B'] + '|\n\n这是已暴露benchmark引导的探索性决策，未作独立盲测或多重比较校正。\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure = out / 'figures'
    figure.mkdir()
    for kind, keys, labels, filename in [('target', ['mgsm_rev2_acc', 'msvamp_acc', 'polymath_DW_ACC_percent'], ['MGSM-Rev2', 'MSVAMP', 'PolyMath DW-ACC'], 'target_gain_vs_training'), ('general', metrics, ['GSM8K flexible', 'MMLU', 'IFEval prompt strict', 'HumanEval+'], 'general_retention_vs_training')]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for ax, lang in zip(axes, langs):
            rs = sorted([r for r in roles if r['language'] == lang], key=lambda r: r['epoch'])
            xs = [r['shifted_supervised_tokens'] / 1000000.0 for r in rs]
            for k, label in zip(keys, labels):
                scale = 1 if k == 'polymath_DW_ACC_percent' else 100
                ys = [(r[k] - rs[0][k]) * scale for r in rs]
                ax.plot(xs, ys, 'o-', label=label)
            ax.axhline(0, color='grey', lw=0.7)
            ax.set_title(lang.upper())
            ax.set_xlabel('Cumulative supervised tokens (million)')
            ax.set_ylabel('Change from Base (percentage points)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.suptitle('Fixed Base / Epoch 1 / Epoch 2; single training seed42')
        fig.tight_layout()
        fig.savefig(figure / (filename + '.png'), dpi=180)
        fig.savefig(figure / (filename + '.svg'))
        plt.close(fig)
    dump(out / 'DONE.json', {'status': 'COMPLETE', 'route_decision': decision['status'], 'model_role_rows': 9, 'unique_weight_models': 7, 'selected_languages': langs, 'analysis_sources': source})
    print(json.dumps(decision, indent=2))
if __name__ == '__main__':
    main()
