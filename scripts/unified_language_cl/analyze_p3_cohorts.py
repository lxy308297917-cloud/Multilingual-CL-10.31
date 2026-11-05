"""Describe the frozen A/B train cohorts; never resample or use model scores."""
from runtime import runtime_path
import collections
import json
import os
import sys
import unicodedata
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, runtime_path('project', 'scripts/unified_language_cl'))
from common import ROOT, DATA, sha, oh, rows, dump
from evaluate import tree
out = Path(runtime_path('results', '')) / 'p3-cohort-composition-v1'
assert not out.exists()
report = {}
table = ['# P3：固定A/B队列组成与预算', '', 'ID在原始训练前协议中预留，本表只读取固定训练数据与协议，不使用模型分数选择或重采样。A是TH_A（与ES_A同源ID），B是预留TH_B（与A ID不重叠），三轨道均为3200例/200步。', '', '| 轨道 | A/B ID重叠 | B中TH文本见于A | 类别分布TV | 类别定义 | B/A输入token | B/A监督token |', '|---|---:|---:|---:|---|---:|---:|']

def normalize(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return unicodedata.normalize('NFKC', text).strip()
for track in ['massive', 'xnli', 'mtop']:
    parent = DATA / ('prepared-v2' if track == 'mtop' else 'prepared-v1') / track
    derived = DATA / 'prepared-disjoint-v1' / track
    p = json.loads((parent / 'protocol.json').read_text())
    b = json.loads((derived / 'protocol.json').read_text())
    assert b['parent_protocol_sha256'] == sha(parent / 'protocol.json')
    a_rows, b_rows = (rows(parent / 'th_train.jsonl'), rows(derived / 'th_train.jsonl'))
    assert [r['id'] for r in a_rows] == p['training_ids']
    assert [r['id'] for r in b_rows] == p['disjoint_reserve_ids'] == b['training_ids']
    assert len(a_rows) == len(b_rows) == len(set(p['training_ids'])) == len(set(b['training_ids'])) == 3200
    overlap = len(set(p['training_ids']) & set(b['training_ids']))
    assert overlap == 0
    label = lambda r: tree(r['answer'])[0] if track == 'mtop' else r['answer']
    ca, cb = (collections.Counter((label(r) for r in a_rows)), collections.Counter((label(r) for r in b_rows)))
    labels = sorted(set(ca) | set(cb))
    tv = sum((abs(ca[k] / 3200 - cb[k] / 3200) for k in labels)) / 2
    a_text = {normalize(r['text']) for r in a_rows}
    seen = sum((normalize(r['text']) in a_text for r in b_rows))
    sa, sb = (p['token_stats']['th:train'], b['token_stats']['th:train'])
    ratios = {'input_after_cutoff': sb['totals'][2] / sa['totals'][2], 'shifted_assistant': sb['totals'][3] / sa['totals'][3]}
    meaning = 'root intent only, not full target trees' if track == 'mtop' else 'classification label'
    report[track] = {'A_rows': 3200, 'B_rows': 3200, 'steps_each': 200, 'A_B_semantic_ID_overlap': overlap, 'B_rows_with_NFKC_stripped_TH_text_in_A': seen, 'label_definition': meaning, 'label_total_variation': tv, 'A_label_counts': dict(sorted(ca.items())), 'B_label_counts': dict(sorted(cb.items())), 'A_token_statistics': sa, 'B_token_statistics': sb, 'B_over_A_tokens': ratios, 'A_ID_order_sha256': oh(p['training_ids']), 'B_ID_order_sha256': oh(b['training_ids']), 'parent_protocol_sha256': sha(parent / 'protocol.json'), 'B_protocol_sha256': sha(derived / 'protocol.json'), 'A_train_json_sha256': sha(parent / 'th_train.jsonl'), 'B_train_json_sha256': sha(derived / 'th_train.jsonl')}
    table.append(f"| {track} | {overlap} | {seen}/3200 | {tv:.5f} | {meaning} | {ratios['input_after_cutoff']:.5f} | {ratios['shifted_assistant']:.5f} |")
table += ['', 'TV为两组类别频率的总变差距离（半L1），仅描述队列组成，不是模型效果。MTOP只对根意图聚合，不公开完整目标树计数，也不拿根意图分布冒充结构难度。', '', 'ID分离不代表语义全新；文本重复不据此删除或换样本。相同步数和样本数也不是相同token/FLOPs预算。正式比较包含直接TH_A和TH_B对照，但不同内容难度及非线性训练效应仍不能完全由差分消去，不能将交互项当作纯语言切换因果。', '', '机器可读文件仅含类别计数、token统计与来源/顺序哈希，不含原始文本、完整树或逐题模型输出。']
out.mkdir()
dump(out / 'audit.json', {'status': 'FROZEN_COHORTS_DESCRIBED_WITHOUT_RESELECTION', 'model_scores_used': False, 'tracks': report})
(out / 'report.md').write_text('\n'.join(table) + '\n')
print(json.dumps({'status': 'FROZEN_COHORTS_DESCRIBED_WITHOUT_RESELECTION', 'tracks': 3}))
