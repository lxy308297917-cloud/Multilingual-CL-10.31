"""Read-only coverage check: every eligible missing cell has a live queue or DONE."""
from pathlib import Path
import csv, datetime, hashlib, io, json, os

R = Path('/path/to/project')
L = Path('/path/to/logs/qwen3-multilingual')
C = R / 'configs/benchmark_5plus5_20261109'
Q = R / 'research/benchmark_5plus5_20261109'
TASK_STAGE = {'Belebele': 'reading', 'XNLI': 'reading', 'XL-Sum': 'summary',
              'Global-MMLU': 'gmmlu', 'GSM8K': 'core', 'IFEval': 'core',
              'HumanEval+': 'code', 'EN MMLU': 'mmlu', 'EN Belebele': 'reading'}

def load(p):
    return json.loads(p.read_text())

def live_pid(pid, expected):
    p = Path('/proc') / str(pid)
    try:
        return p.stat().st_uid == os.getuid() and expected in (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, PermissionError):
        return False

def find_live(expected):
    return any(live_pid(p.name, expected) for p in Path('/proc').iterdir() if p.name.isdigit())

def main():
    models = load(C / 'models.json')
    deferred = set(load(C / 'deferred_models.json'))
    plans = load(C / 'priority_evaluation_plan.json')
    queues = {}
    for family in ['fa', 'fft']:
        plan = plans[family]
        state = load(L / plan['log_family'] / 'evaluation-schedule.json')
        pending = set(map(tuple, state['pending']))
        active = state.get('running', state.get('active', []))
        pending.update((x[plan['key']], x['stage']) for x in active)
        queues[family] = dict(pending=pending, live=find_live('priority_evaluations_v2.py --family ' + family), root=L / plan['log_family'] / 'evaluation')
    for family, state_name, command in [('supplement', 'schedule.json', 'benchmark_5plus5_20261109/schedule_v3.py'), ('general', 'general-schedule.json', 'benchmark_5plus5_20261109/general_schedule.py')]:
        state = load(L / 'benchmark-5plus5-20261109' / state_name)
        assert 'pid' in state or state.get('status') == 'ALL_TERMINAL_PENDING_RAW_AUDIT', state
        queues[family] = dict(pending=set(map(tuple, state['pending'])), live=live_pid(state['pid'], command) if 'pid' in state else False, root=L / 'benchmark-5plus5-20261109')
    data = (Q / 'scores_5plus5.csv').read_bytes()
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    assert {x['model'] for x in rows} == set(models)
    cells, uncovered, deferred_rows = [], [], []
    for row in rows:
        name = row['model']
        if name in deferred:
            assert row['execution_status'] == 'USER_DEFERRED'
            deferred_rows.append(name)
            continue
        for task in ['SIB-200', *TASK_STAGE]:
            if row[task] != '':
                cells.append(dict(model=name, task=task, status='PRESENT_IN_AUDITED_PROJECTION'))
                continue
            if task == 'SIB-200':
                family, stage = 'supplement', 'sib200'
            elif name.endswith('_historical_lora'):
                stage = TASK_STAGE[task]
                family = 'supplement' if stage == 'gmmlu' else 'general'
            else:
                family = 'fa' if models[name].get('source_family') == 'function_aware_formal_20261109' or name.endswith('_Base') else 'fft'
                stage = TASK_STAGE[task]
            queue = queues[family]
            out = queue['root'] / name / stage
            if (out / 'DONE.json').exists():
                status = 'DONE_AWAITING_PROJECTION_REFRESH'
            elif queue['live'] and (name, stage) in queue['pending']:
                status = 'COVERED_BY_LIVE_QUEUE'
            else:
                status = 'UNCOVERED'
                uncovered.append(dict(model=name, task=task, family=family, stage=stage))
            cells.append(dict(model=name, task=task, stage=stage, family=family, status=status))
    assert len(cells) == (len(models) - len(deferred)) * 10
    assert set(deferred_rows) == deferred
    xroot = L / 'xquad-20261109'
    xstate = load(xroot / 'schedule.json')
    xmodels = load(R / 'configs/xquad_20261109/models.json')
    xlive = live_pid(xstate['pid'], 'xquad_20261109/schedule.py')
    xcells = []
    for name in xmodels:
        if (xroot / name / 'DONE.json').exists():
            status = 'DONE_SUBJECT_TO_RAW_AUDIT'
        elif xlive and name in xstate['pending']:
            status = 'COVERED_BY_LIVE_QUEUE'
        else:
            status = 'UNCOVERED'
            uncovered.append(dict(model=name, task='XQuAD'))
        xcells.append(dict(model=name, status=status))
    report = dict(at=datetime.datetime.now().astimezone().isoformat(),
                  status='ALL_ELIGIBLE_MISSING_CELLS_COVERED' if not uncovered else 'UNCOVERED_CELLS_REQUIRE_ACTION',
                  scope='Queue coverage only. Present scores depend on separate raw audits; live queues do not prove successful completion.',
                  score_table_sha256=hashlib.sha256(data).hexdigest(),
                  eligible_models=len(models)-len(deferred), required_cells=len(cells),
                  present_cells=sum(x['status']=='PRESENT_IN_AUDITED_PROJECTION' for x in cells),
                  deferred_models=sorted(deferred), queue_live={k:v['live'] for k,v in queues.items()},
                  cells=cells, xquad_required=len(xcells), xquad_queue_live=xlive, xquad_cells=xcells, uncovered=uncovered,
                  new_gpu_jobs=0, new_training_jobs=0)
    (Q / 'coverage_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','eligible_models','required_cells','present_cells','xquad_required','uncovered']}))
    assert not uncovered, uncovered

if __name__ == '__main__':
    main()
