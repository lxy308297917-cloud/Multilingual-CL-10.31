"""Matched fixed dev64 readout audit, separate from formal benchmark effects."""
from runtime import runtime_path
import os, sys, json, collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, runtime_path('project', 'scripts/unified_language_cl'))
from common import ROOT, DATA, rows, sha, dump
from evaluate import metrics
out = Path(runtime_path('results', '')) / 'first_batch_readout_v1'
assert not out.exists()
out.mkdir()
report = {}
table = ['# 固定开发材料上的读出诊断', '', '本表不是正式benchmark成绩，不用于选择checkpoint。生成ACC与候选分数ACC都在相同64题/语言计算；候选用合法标签token log-prob之和，未做长度归一化。', '', '| 轨道 | 模型 | 语言 | n | 生成ACC | 候选ACC | 生成非法标签率 |', '|---|---|---|---:|---:|---:|---:|']
for t in ['massive', 'xnli', 'sib200']:
    root = Path(runtime_path('results', '')) / '' / t
    aliases = json.loads((root / 'run_aliases.json').read_text()) if (root / 'run_aliases.json').exists() else {}
    report[t] = {}
    labels = json.loads((DATA / 'prepared-v1' / t / 'protocol.json').read_text())['labels']
    for name in ['base', 'single_es', 'single_th', 'seq_es_th']:
        p = root / aliases.get(name + '-dev-v1', name + '-dev-v1')
        if not (p / 'DONE.json').exists() or not (p / 'candidate_predictions.jsonl').exists():
            continue
        gen = {(r['lang'], r['id']): r for r in rows(p / 'predictions.jsonl')}
        cand = rows(p / 'candidate_predictions.jsonl')
        by = collections.defaultdict(list)
        for r in cand:
            assert (r['lang'], r['id']) in gen
            by[r['lang']].append(r)
        model = {}
        for l, rs in by.items():
            assert len(rs) == len({r['id'] for r in rs}) == 64
            gr = [gen[l, r['id']] for r in rs]
            m = metrics(gr)
            invalid = sum((r['output'].strip() not in labels for r in gr)) / 64
            c = sum((r['correct'] for r in rs)) / 64
            model[l] = {'n': 64, 'generation_accuracy': m['score'], 'candidate_accuracy': c, 'generation_invalid_label_rate': invalid, 'ids_sha256': __import__('hashlib').sha256(json.dumps(sorted((r['id'] for r in rs))).encode()).hexdigest()}
            table.append(f"| {t} | {name} | {l} | 64 | {m['score'] * 100:.2f} | {c * 100:.2f} | {invalid * 100:.2f} |")
        report[t][name] = {'langs': model, 'generation_sha256': sha(p / 'predictions.jsonl'), 'candidate_sha256': sha(p / 'candidate_predictions.jsonl')}
    for l in ['es', 'th', 'fr', 'en']:
        assert len({v['langs'][l]['ids_sha256'] for v in report[t].values()}) <= 1
math = {}
root = Path(runtime_path('results', '')) / 'mathoctopus'
aliases = json.loads((root / 'run_aliases.json').read_text())
for name in ['base', 'single_es', 'single_th', 'seq_es_th']:
    p = root / aliases.get(name + '-test-v1', name + '-test-v1')
    rs = rows(p / 'predictions.jsonl')
    groups = collections.defaultdict(list)
    for r in rs:
        groups[r['lang'] + ':' + r['benchmark']].append(r)
    math[name] = {}
    for key, group in groups.items():
        m = metrics(group)
        math[name][key] = {**m, 'mean_generated_tokens': sum((r['generated_tokens'] for r in group)) / len(group), 'format_caveat': 'Numerical answer ACC is primary; strict #### is a separate diagnostic. No test-based parser or checkpoint changes.'}
dump(out / 'classification_matched_dev64.json', report)
dump(out / 'math_formal_output_audit.json', math)
(out / 'matched_dev64.md').write_text('\n'.join(table) + '\n')
print(json.dumps({'status': 'DONE', 'classification_tracks': list(report), 'math_models': list(math)}))
