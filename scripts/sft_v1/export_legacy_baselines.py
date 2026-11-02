"""Export completed historical MURI Finals, separate from Aya/native protocols."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

LOG = Path('/path/to/workspace/logs/qwen3-multilingual/evaluation')
BASE = dict(belebele='base17b_belebele_20261031_v1', mmlu='base17b_mmlu_20261031_r2',
            gmmlu='base17b_gmmlu_20261031_r2', ifeval='base17b_ifeval_20261031_r2',
            gsm8k='base17b_gsm8k_20261031_r2', flores='base17b_flores_20261031_r2')
METHODS = [
    ('base', '原模型', None, '未训练', None),
    ('lora1250', 'LoRA一轮1250', 1250, '固定一轮；10000条', 'muri_fixed_epoch1_lora_{task}_20261101_v1'),
    ('lora2500', 'LoRA两轮2500', 2500, '两轮；20000条；不同预算', 'muri_terminal_epoch2_lora_{task}_20261101_v1'),
    ('full_r2', 'Full1250旧r2', 1250, '恢复实际LR偏差；补充结果', 'muri_fixed_epoch1_full_{task}_20261101_v1'),
    ('full_r3', 'Full1250正确r3', 1250, '原cosine日程接续已修正', 'muri_fixed_epoch1_full_{task}_20261102_lrfix_r3'),
]
COLUMNS = [
    ('EN阅读 ACC', 'belebele_eng_Latn_locked', 'acc,none', 900),
    ('SW阅读 ACC', 'belebele_swh_Latn_locked', 'acc,none', 900),
    ('EN知识 ACC', 'mmlu_locked', 'acc,none', 14042),
    ('SW知识 ACC', 'global_mmlu_full_sw_locked', 'acc,none', 14042),
    ('IFEval提示strict', 'ifeval_locked', 'prompt_level_strict_acc,none', 541),
    ('IFEval指令strict', 'ifeval_locked', 'inst_level_strict_acc,none', 541),
    ('GSM8K strict', 'gsm8k_locked', 'exact_match,strict-match', 1319),
    ('GSM8K flexible辅助', 'gsm8k_locked', 'exact_match,flexible-extract', 1319),
    ('EN→SW chrF++', 'eng_Latn_to_swh_Latn', 'chrF++', 500),
    ('SW→EN chrF++', 'swh_Latn_to_eng_Latn', 'chrF++', 500),
]
SELECTED = {(task, metric): count for _, task, metric, count in COLUMNS}
SELECTED.update({('ifeval_locked', metric): 541 for metric in
                 ['prompt_level_loose_acc,none', 'inst_level_loose_acc,none']})


def collect(method, step, template):
    data, identities = {}, []
    for task in BASE:
        run = LOG / (BASE[task] if method == 'base' else template.format(task=task))
        assert (run / 'exit_code.txt').read_text().strip() == '0', run
        if method != 'base':
            meta = json.loads((run / 'run-start.json').read_text())
            model = meta['verified_model']
            assert model['status'] == 'validated' and model['artifact_step'] == step
            assert meta['exposure']['samples'] == (20000 if step == 2500 else 10000)
            identities.append((model['model_dir'], model['weights']))
            audit = 'accuracy-audit.json' if task in ('belebele', 'mmlu', 'gmmlu') else 'score-audit.json'
            assert (run / audit).exists(), run
        if task == 'flores':
            source = run / 'results/summary.json'
            raw = json.loads(source.read_text())
            values = {(name, 'chrF++'): (score['chrF++'], score['samples']) for name, score in raw.items()}
        else:
            files = list(run.glob('results/**/results_*.json'))
            assert len(files) == 1, run
            source = files[0]
            raw = json.loads(source.read_text())['results']
            values = {(name, metric): (scores[metric] * 100, scores['sample_len'])
                      for name, scores in raw.items() for metric in scores if (name, metric) in SELECTED}
        for key, (score, count) in values.items():
            assert key in SELECTED and count == SELECTED[key] and 0 <= score <= 100
            assert key not in data
            data[key] = (score, count, str(source))
    assert set(data) == set(SELECTED)
    assert not identities or all(item == identities[0] for item in identities)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--json-output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.json_output.exists()
    book = Workbook()
    table = book.active
    table.title = '旧MURI固定终点'
    table.append(['模型', '结果身份'] + [name for name, *_ in COLUMNS])
    evidence = book.create_sheet('逐项证据')
    evidence.append(['方法', '任务', '指标', '题目/提示数', '分数0–100', '原始结果路径'])
    rows = []
    for method, label, step, note, template in METHODS:
        values = collect(method, step, template)
        table.append([label, note] + [values[task, metric][0] for _, task, metric, _ in COLUMNS])
        for (task, metric), (score, count, source) in sorted(values.items()):
            evidence.append([label, task, metric, count, score, source])
            row = dict(method=method, label=label, artifact_step=step, task=task, metric=metric,
                       score_0_to_100=score, question_or_prompt_count=count, official_result_file=source)
            if task == 'ifeval_locked' and metric.startswith('inst_'):
                row['instruction_denominator'] = 834
            rows.append(row)
    notes = book.create_sheet('设置与限制')
    for row in [
        ['项目', '说明'],
        ['数据与预算', 'MURI SW10000条/轮；每轮3805651输入、3230101移位回答监督token；seed42。'],
        ['Full', 'LR5e-5；单卡ZeRO2 CPUAdam；一轮1250步，原5000步cosine日程前缀；313步接续。'],
        ['LoRA', 'rank8/alpha16、七类Attention/MLP投影、LR1e-4；1250一轮，2500两轮。'],
        ['身份区别', '旧Full r2恢复实际LR固定：保留为补充。r3实际LR逐步与scheduler一致；BF16接续不承诺逐位重放。'],
        ['评测协议', '沿用原HF协议；IFEval Base66.7283、GSM8K Base43.8211；不与新Aya原生vLLM分数混用。'],
        ['指标', 'ACC/EM/通过率为0–100百分数；chrF++为原刻度；IFEval541提示/834条指令。'],
        ['格式限制', 'GSM8K strict为主、flexible辅助；两者均保留。选择题选项偏置可能影响ACC，不做因果知识损失归因。'],
        ['选模', '固定终点；不因英语分数拒绝baseline。旧313/939早停或Dev-best点保留历史，不代替Final。'],
        ['比较范围', 'LoRA两轮与一轮预算不同。已见正式测试后的补充比较，不能用本表再选新方案超参数。'],
    ]:
        notes.append(row)
    for sheet in book:
        sheet.freeze_panes = 'C2' if sheet == table else 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        sheet.sheet_view.showGridLines = False
        for cell in sheet[1]:
            cell.fill = PatternFill('solid', fgColor='17365D')
            cell.font = Font(color='FFFFFF', bold=True)
        for line in sheet.iter_rows(min_row=2):
            for cell in line:
                cell.alignment = Alignment(vertical='top', wrap_text=True)
                if isinstance(cell.value, float):
                    cell.number_format = '0.00'
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.page_setup.orientation = 'landscape'
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A3
        sheet.page_setup.fitToWidth, sheet.page_setup.fitToHeight = 1, 0
        sheet.print_title_rows = '1:1'
    for i in range(1, 13):
        table.column_dimensions[get_column_letter(i)].width = 20 if i > 2 else 28
    table.print_area = table.dimensions
    for i, width in enumerate([24, 46, 35, 16, 16, 100], 1):
        evidence.column_dimensions[get_column_letter(i)].width = width
    notes.column_dimensions['A'].width, notes.column_dimensions['B'].width = 22, 125
    for line in notes.iter_rows(min_row=2):
        notes.row_dimensions[line[0].row].height = 40
    args.output.parent.mkdir(parents=True, exist_ok=True)
    book.save(args.output)
    args.json_output.write_text(json.dumps(dict(snapshot_at=datetime.now().astimezone().isoformat(),
        scope='Complete old MURI Finals only; separate original HF protocol; all12 metrics/model retained',
        rows=rows), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'metrics': len(rows), 'models': len(METHODS), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
