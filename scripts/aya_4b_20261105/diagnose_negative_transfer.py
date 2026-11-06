"""Read-only output, label and option-scoring diagnosis; no test-based tuning."""
from profile_paths import resolve
import argparse, json, hashlib, re, collections, statistics, csv, zipfile
from pathlib import Path
L = Path(resolve('LOGS'))
D = Path(resolve('DATA'))

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def rows(p):
    return [json.loads(x) for x in p.open()]

def norm(s):
    return ' '.join(s.casefold().split())

def dump(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--output', type=Path, required=True)
    args = a.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    translation = []
    unrun = []
    for role in ['base', 'fft', 'lora', 'ssu']:
        for direction in ['en_sw', 'sw_en']:
            root = L / (role + '-generation-v2') / ('flores200_' + direction + '_full_devtest_v2')
            if not (root / 'DONE.json').exists():
                unrun.append(role + '/' + direction)
                continue
            xs = rows(root / 'predictions.jsonl')
            assert len(xs) == 1012
            rep = caps = empty = copy = 0
            length_ratios = []
            both = 0
            for x in xs:
                words = re.findall('\\w+', x['prediction'].casefold())
                grams = collections.Counter((tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))))
                repeated = len(words) >= 24 and max(grams.values(), default=0) >= 5 and (len(grams) / max(1, sum(grams.values())) < 0.5)
                capped = bool(x['generation_hit_cap'])
                rep += repeated
                caps += capped
                both += repeated and capped
                empty += not x['prediction'].strip()
                copy += norm(x['prediction']) == norm(x['source'])
                length_ratios.append(len(x['prediction']) / max(1, len(x['reference'])))
            translation.append({'model': role, 'direction': direction.replace('_', '->'), 'samples': len(xs), 'cap_count': caps, 'repetitive_4gram_count': rep, 'both_cap_and_repetitive': both, 'empty_count': empty, 'exact_source_copy_count': copy, 'median_prediction_reference_character_ratio': statistics.median(length_ratios), 'mean_generated_tokens': statistics.mean((x['generated_tokens'] for x in xs)), 'predictions_sha256': digest(root / 'predictions.jsonl')})
    dump(out / 'translation_output_diagnosis.json', {'status': 'PARTIAL' if unrun else 'COMPLETE', 'rows': translation, 'not_yet_available': unrun, 'repetition_definition': 'At least24 Unicode-word tokens, max count of one4gram>=5, distinct4grams/total4grams<0.5. Heuristic only; original predictions and corpus scores unchanged.', 'causal_claim': 'Repetition and cap coincide with poor translation; this alone does not identify whether decoding, training data or model changes cause it.'})
    train = rows(D / 'mixture-v1/train6400.jsonl')
    provenance = rows(D / 'mixture-v1/train6400.provenance.jsonl')
    assert len(train) == len(provenance) == 6400
    with zipfile.ZipFile(D / 'raw/globalvoices-en-sw-v2018q4.zip') as z:
        er = z.read(next((n for n in z.namelist() if n.endswith('.en')))).decode().splitlines()
        sr = z.read(next((n for n in z.namelist() if n.endswith('.sw')))).decode().splitlines()
    counts = collections.Counter()
    groups = collections.defaultdict(set)
    for t, p in zip(train, provenance):
        if p['task'] not in ['en_to_sw', 'sw_to_en']:
            continue
        i = int(p['source_id'])
        en, sw = (er[i], sr[i])
        forward = p['task'] == 'en_to_sw'
        source, target = (en, sw) if forward else (sw, en)
        prompt = 'Translate the following ' + ('English text into Swahili' if forward else 'Swahili text into English') + '. Output only the translation.\n\n' + source
        assert t['instruction'] == prompt.strip() and t['output'] == target.strip(), i
        counts[p['task']] += 1
        groups[i].add(p['task'])
    assert counts == {'en_to_sw': 640, 'sw_to_en': 640} and len(groups) == 640 and all((len(v) == 2 for v in groups.values()))
    from datasets import load_from_disk
    tokenized = load_from_disk(str(D / 'mixture-v1/train6400_tokenized'))['train']
    tails = collections.Counter((tuple(x['labels'][-2:]) for x in tokenized))
    assert tails == {(151645, 198): 6400}, tails
    dump(out / 'training_direction_and_termination_audit.json', {'status': 'VERIFIED', 'translation_samples': dict(counts), 'independent_parallel_groups': len(groups), 'all_direction_prompts_and_targets_match_original_parallel_lines': True, 'rows_supervising_im_end_then_newline': 6400, 'ending_tokens': [151645, 198], 'translation_eval_stop_tokens': [151645, 151643], 'note': 'Last token alone is newline198, not missing EOS; im_end151645 is correctly supervised immediately before newline. No missing-end-label explanation was established.', 'source_sha256': {'parallel_zip': digest(D / 'raw/globalvoices-en-sw-v2018q4.zip'), 'train': digest(D / 'mixture-v1/train6400.jsonl'), 'provenance': digest(D / 'mixture-v1/train6400.provenance.jsonl')}})
    models = {}
    checks = []
    for role in ['base', 'fft', 'lora', 'ssu']:
        root = L / (role + '-harness-v1/gmmlu')
        if not (root / 'DONE.json').exists():
            continue
        values = {}
        files = sorted(root.rglob('samples_*.jsonl'))
        assert len(files) == 57, (role, len(files))
        for f in files:
            for x in rows(f):
                doc = x['doc']
                likelihoods = [float(v[0]) for v in x['filtered_resps']]
                assert len(likelihoods) == 4
                gold = 'ABCD'.index(doc['answer'])
                pred = max(range(4), key=likelihoods.__getitem__)
                correct = int(pred == gold)
                assert correct == x['acc']
                key = (doc['subject'], str(doc['sample_id']))
                identity = hashlib.sha256(json.dumps({k: doc[k] for k in ['question', 'option_a', 'option_b', 'option_c', 'option_d', 'answer']}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                assert key not in values
                values[key] = {'identity': identity, 'correct': correct, 'category': doc['subject_category']}
        assert len(values) == 14042
        models[role] = values
        checks.append({'model': role, 'subjects': len(files), 'questions': len(values), 'four_likelihood_argmax_and_gold_mapping_match_all_saved_ACC': True})
    comparisons = []
    for role, values in models.items():
        if role == 'base':
            continue
        base = models['base']
        assert base.keys() == values.keys() and all((base[k]['identity'] == values[k]['identity'] for k in base))
        for grouping in ['subject', 'category']:
            groups = collections.defaultdict(list)
            for k, v in values.items():
                groups[k[0] if grouping == 'subject' else v['category']].append((base[k]['correct'], v['correct']))
            for group, pairs in sorted(groups.items()):
                b = sum((x for x, y in pairs))
                v = sum((y for x, y in pairs))
                comparisons.append({'model': role, 'grouping': grouping, 'group': group, 'samples': len(pairs), 'base_correct': b, 'final_correct': v, 'base_ACC': 100 * b / len(pairs), 'final_ACC': 100 * v / len(pairs), 'delta_pp': 100 * (v - b) / len(pairs), 'wrong_to_right': sum((not x and y for x, y in pairs)), 'right_to_wrong': sum((x and (not y) for x, y in pairs))})
    dump(out / 'knowledge_likelihood_audit.json', {'status': 'PARTIAL' if len(models) < 4 else 'COMPLETE', 'models_checked': checks, 'scoring': 'Four-option likelihood ACC; generation letter parsing is not involved. All actual questions, options, gold labels and sample identities compared locally; raw content excluded here.', 'models_not_yet_available': [r for r in ['base', 'fft', 'lora', 'ssu'] if r not in models]})
    with (out / 'knowledge_by_subject_and_category.csv').open('w', newline='') as f:
        fields = ['model', 'grouping', 'group', 'samples', 'base_correct', 'final_correct', 'base_ACC', 'final_ACC', 'delta_pp', 'wrong_to_right', 'right_to_wrong']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(comparisons)
    text = ['# 翻译与知识瓶颈：只读诊断', '', '此诊断不修改正式输出、指标、训练配方或checkpoint。未完成方法不以零分填充。', '', '## 翻译输出异常', '', '|模型|方向|样本|达到512上限|重复4gram启发式|输出/参考字符数中位比|', '|---|---|---:|---:|---:|---:|']
    for r in translation:
        text.append(f"|{r['model']}|{r['direction']}|{r['samples']}|{r['cap_count']}|{r['repetitive_4gram_count']}|{r['median_prediction_reference_character_ratio']:.2f}|")
    text += ['', 'EN→SW存在重复和长度上限问题，原模型已有，适配后加重。SW→EN没有相同规模的问题。统计关联不构成解码或训练的因果证明。', '', '## 已核验的工程因素', '', '- 全1280条翻译训练指令方向与原始640句对匹配，未发现反向标签/句对错位。', '- 全6400条训练监督均包含im_end151645及其后newline198；评测停止token包含151645。不能把问题直接归因于漏掉EOS监督。', '- 已完成知识测评的全部四选项likelihood argmax与原ACC一致；没有生成字母解析环节。逐学科/大类变化见CSV。', '', '## 下一步开发验证', '', '在预先冻结、排除few-shot来源和已训练平行句组的开发集上，比较原greedy与官方建议non-thinking采样（temperature0.7、top_p0.8、top_k20、min_p0）。同模型、同提示、同样本、同输出上限、固定seed。只作独立协议诊断，不覆盖本轮正式greedy表，也不在正式test上挑解码参数。', '', '官方模型卡：https://huggingface.co/Qwen/Qwen3-4B#best-practices 。官方non-thinking推荐值支持该有限对照，但其明确的greedy重复警告描述的是thinking场景，不能据此直接断言当前non-thinking问题的原因。', '', '新增开发GPU对照尚未运行（NOT_RUN）；先核查空闲GPU，保持既有正式测评优先。']
    (out / 'diagnosis.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({'status': 'DIAGNOSTIC_EVIDENCE_EXPORTED', 'translation_groups': len(translation), 'knowledge_models': list(models), 'subject_and_category_rows': len(comparisons), 'output': str(out)}))
if __name__ == '__main__':
    main()
