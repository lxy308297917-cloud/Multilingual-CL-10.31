"""Complete the predeclared matched-dev64 diagnosis; no generation or selection."""
from runtime import runtime_path
import os
import sys
import json
import math
from collections import defaultdict
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, runtime_path('project', 'scripts/unified_language_cl'))
from common import ROOT, DATA, rows, sha, oh, dump
from evaluate import metrics
out = Path(runtime_path('results', '')) / 'classification_readout_complete_v2'
assert not out.exists(), out
result = {}
table = ['# 分类任务：固定开发64题读出复核', '', '三个轨道、四个固定模型、四种语言均使用原先冻结的64题。生成与候选评分在同题上比较。候选使用全部合法标签的token log-prob之和，不做长度归一化。这里只诊断读出与标签格式，不替代正式全量测试ACC，不用于调参或选模型。', '', '| 轨道 | 模型 | 语言 | n | 生成ACC% | 候选ACC% | 非法标签率% |', '|---|---|---|---:|---:|---:|---:|']
for track in ['massive', 'xnli', 'sib200']:
    root = Path(runtime_path('results', '')) / '' / track
    aliases = json.loads((root / 'run_aliases.json').read_text()) if (root / 'run_aliases.json').exists() else {}
    labels = json.loads((DATA / 'prepared-v1' / track / 'protocol.json').read_text())['labels']
    result[track] = {}
    expected_ids = {}
    expected_prompts = {}
    for model in ['base', 'single_es', 'single_th', 'seq_es_th']:
        path = root / aliases.get(model + '-dev-v1', model + '-dev-v1')
        assert (path / 'DONE.json').exists()
        generated = rows(path / 'predictions.jsonl')
        done = json.loads((path / 'DONE.json').read_text())
        assert done['predictions_sha256'] == sha(path / 'predictions.jsonl')
        gen = {(r['lang'], r['id']): r for r in generated}
        assert len(gen) == len(generated)
        candidates = rows(path / 'candidate_predictions.jsonl')
        groups = defaultdict(list)
        for row in candidates:
            assert len(row['scores']) == len(labels) and all((math.isfinite(v) for v in row['scores']))
            pred = labels[max(range(len(labels)), key=lambda j: row['scores'][j])]
            assert pred == row['prediction'] and row['correct'] == (pred == row['answer'])
            assert row['answer'] == gen[row['lang'], row['id']]['answer']
            groups[row['lang']].append(row)
        assert set(groups) == {'es', 'th', 'fr', 'en'}
        langs = {}
        for lang, rs in sorted(groups.items()):
            ids = sorted((r['id'] for r in rs))
            assert len(ids) == len(set(ids)) == 64
            frozen = sorted([r for r in generated if r['lang'] == lang], key=lambda r: oh([42, r['id']]))[:64]
            assert ids == sorted((r['id'] for r in frozen))
            prompts = {r['id']: r['prompt_token_sha256'] for r in frozen}
            if lang not in expected_ids:
                expected_ids[lang], expected_prompts[lang] = (ids, prompts)
            assert ids == expected_ids[lang] and prompts == expected_prompts[lang]
            paired = [gen[lang, r['id']] for r in rs]
            generation = metrics(paired)['score'] * 100
            candidate = sum((r['correct'] for r in rs)) / 64 * 100
            invalid = sum((r['output'].strip() not in labels for r in paired)) / 64 * 100
            langs[lang] = {'samples': 64, 'generation_accuracy_percent': generation, 'candidate_accuracy_percent': candidate, 'invalid_label_percent': invalid, 'ids_sha256': oh(ids), 'prompt_hash_mapping_sha256': oh(prompts)}
            table.append(f'| {track} | {model} | {lang} | 64 | {generation:.2f} | {candidate:.2f} | {invalid:.2f} |')
        result[track][model] = {'langs': langs, 'generation_sha256': sha(path / 'predictions.jsonl'), 'candidate_sha256': sha(path / 'candidate_predictions.jsonl')}
table += ['', '候选集合和任务提示已透露任务结构，候选得分也包含标签token偏好；其变化不能单独证明新增语言语义知识。64题的差异不等于全量正式测试差异。仅seed42，未新增GPU计算，未改变评分器或原有结果。']
out.mkdir()
dump(out / 'effects.json', {'status': 'COMPLETE_ALL_48_MATCHED_DEV_GROUPS', 'groups': result, 'selection_or_training_changes': False, 'new_gpu_generation': False})
(out / 'table.md').write_text('\n'.join(table) + '\n')
print(json.dumps({'status': 'COMPLETE_ALL_48_MATCHED_DEV_GROUPS', 'output': str(out)}))
