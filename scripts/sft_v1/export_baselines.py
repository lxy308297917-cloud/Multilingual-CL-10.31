"""Export only completed frozen-protocol scores; never select checkpoints."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.comments import Comment
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.pagebreak import Break

LOG = Path('/path/to/workspace/logs/qwen3-multilingual/sft-v1')
OLD = LOG.parent/'evaluation'
METHODS = [('base','原模型'),('full','FFT'),('lora','LoRA r16'),('ssu','SSU-Wanda')]
# (column title, task, official metric); all displayed on 0-100 scale.
TARGET = [('SW阅读 ACC','belebele_swh_Latn_locked','acc,none'),
 ('SW知识 ACC','global_mmlu_full_sw_locked','acc,none'),
 ('SW ProX EM†','sft_v1_mmlu_prox_sw','exact_match,custom-extract'),
 ('SW数学 strict','sft_v1_mgsm_native_cot_sw','exact_match,strict-match'),
 ('SW数学 flexible‡','sft_v1_mgsm_native_cot_sw','exact_match,flexible-extract'),
 ('SW指令 提示strict','sft_v1_xifeval_sw','prompt_level_strict_acc,none'),
 ('SW指令 指令strict','sft_v1_xifeval_sw','inst_level_strict_acc,none'),
 ('EN→SW chrF++','eng_Latn_to_swh_Latn','chrF++')]
GENERAL = [('EN阅读 ACC','belebele_eng_Latn_locked','acc,none'),
 ('EN知识 ACC','mmlu_locked','acc,none'),
 ('EN ProX EM','sft_v1_mmlu_prox_en','exact_match,custom-extract'),
 ('IFEval 提示strict','sft_v1_ifeval','prompt_level_strict_acc,none'),
 ('IFEval 指令strict','sft_v1_ifeval','inst_level_strict_acc,none'),
 ('GSM8K strict','sft_v1_gsm8k','exact_match,strict-match'),
 ('GSM8K flexible‡','sft_v1_gsm8k','exact_match,flexible-extract'),
 ('SW→EN chrF++','swh_Latn_to_eng_Latn','chrF++')]


def collect(method, seed=42):
    if seed != 42:
        return collect_replication(method, seed)
    data = {}
    for group in ['prox-en','prox-sw','other-core']:
        suffix=group if method=='base' else ('other' if group=='other-core' else group)
        run=LOG/(('base-core-' if method=='base' else 'final-'+method+'-core-')+suffix)
        if not (run/'exit_code.txt').exists() or (run/'exit_code.txt').read_text().strip()!='0':continue
        result=list(run.glob('results/**/results_*.json'));assert len(result)==1
        for task, scores in json.loads(result[0].read_text())['results'].items():
            for metric,score in scores.items():
                if ',' in metric and 'stderr' not in metric and isinstance(score,(int,float)):
                    data[task,metric]=(score*100,scores['sample_len'],str(result[0]))
    for task in ['belebele','mmlu','gmmlu','flores']:
        base_name={'belebele':'base17b_belebele_20261031_v1','mmlu':'base17b_mmlu_20261031_r2','gmmlu':'base17b_gmmlu_20261031_r2','flores':'base17b_flores_20261031_r2'}[task]
        run=OLD/base_name if method=='base' else LOG/('complementary-'+method+'-'+task+'-v1')
        if not (run/'exit_code.txt').exists() or (run/'exit_code.txt').read_text().strip()!='0':continue
        if task=='flores':
            f=run/'results/summary.json'
            for key,value in json.loads(f.read_text()).items():data[key,'chrF++']=(value['chrF++'],value['samples'],str(f))
        else:
            result=list(run.glob('results/**/results_*.json'));assert len(result)==1
            for key,value in json.loads(result[0].read_text())['results'].items():
                for metric,score in value.items():
                    if ',' in metric and 'stderr' not in metric and isinstance(score,(int,float)):
                        data[key,metric]=(score*100,value['sample_len'],str(result[0]))
    return data


def collect_replication(method, seed):
    """Publish full, validated task groups while other groups remain pending."""
    assert seed in (43, 44) and method in ('full', 'lora', 'ssu')
    control = LOG / f'seed{seed}-{method}-evaluation'
    start = control / 'run-start.json'
    if not start.exists():
        return {}
    model = json.loads(start.read_text())['model']
    assert model['artifact_step'] == 400 and model['kind'] == ('lora' if method == 'lora' else 'full')
    plan = Path(__file__).resolve().parents[2] / f'configs/sft_v1/replication_seed{seed}/evaluation_commands.json'
    spec = json.loads(plan.read_text())['methods'][method]
    assert model['model_dir'] == spec['model_dir'] and len(spec['entries']) == 7
    counts = {'sft_v1_gsm8k': 1319, 'sft_v1_ifeval': 541, 'sft_v1_mgsm_native_cot_sw': 250,
              'sft_v1_xifeval_sw': 429, 'sft_v1_mmlu_prox_en': 11759, 'sft_v1_mmlu_prox_sw': 11759,
              'belebele_eng_Latn_locked': 900, 'belebele_swh_Latn_locked': 900, 'mmlu_locked': 14042,
              'global_mmlu_full_sw_locked': 14042, 'eng_Latn_to_swh_Latn': 500, 'swh_Latn_to_eng_Latn': 500}
    tasks = {'other-core': ['sft_v1_gsm8k', 'sft_v1_ifeval', 'sft_v1_mgsm_native_cot_sw', 'sft_v1_xifeval_sw'],
             'prox-en': ['sft_v1_mmlu_prox_en'], 'prox-sw': ['sft_v1_mmlu_prox_sw'],
             'belebele': ['belebele_eng_Latn_locked', 'belebele_swh_Latn_locked'], 'mmlu': ['mmlu_locked'],
             'gmmlu': ['global_mmlu_full_sw_locked'], 'flores': ['eng_Latn_to_swh_Latn', 'swh_Latn_to_eng_Latn']}
    selected = {(task, metric) for _, task, metric in TARGET + GENERAL}
    selected |= {(task, metric) for task in ['sft_v1_ifeval', 'sft_v1_xifeval_sw']
                 for metric in ['prompt_level_loose_acc,none', 'inst_level_loose_acc,none']}
    assert len(selected) == 20
    data = {}
    for entry in spec['entries']:
        run, task = Path(entry['run']), entry['task']
        code = run / 'exit_code.txt'
        if not code.exists() or code.read_text().strip() != '0':
            continue
        if task == 'flores':
            proof = run / 'base-input-identity.json'
        else:
            name = ('paired-' if entry['backend'] == 'core' else 'paired-hf-') + task + '.json'
            proof = control / name
        if not proof.exists():
            continue
        try:
            verified = json.loads(proof.read_text())
        except json.JSONDecodeError:
            continue  # A concurrently written proof is not yet publishable.
        if task == 'flores':
            assert verified['status'] == 'same500 actual inputs/references/prompts per direction'
            assert json.loads((run / 'score-audit.json').read_text())['status'] == 'validated'
        else:
            assert verified['status'].startswith('validated')
        metadata = (json.loads((run / 'model-validated.json').read_text()) if entry['backend'] == 'core'
                    else json.loads((run / 'run-start.json').read_text())['model'])
        assert metadata['weights'] == model['weights'] and metadata['model_dir'] == model['model_dir']
        expected = {key for key in selected if key[0] in tasks[task]}
        group = {}
        if task == 'flores':
            result = run / 'results/summary.json'
            for name, scores in json.loads(result.read_text()).items():
                group[name, 'chrF++'] = (scores['chrF++'], scores['samples'], str(result))
        else:
            results = list(run.glob('results/**/results_*.json'))
            assert len(results) == 1, run
            result = results[0]
            for name, scores in json.loads(result.read_text())['results'].items():
                for metric, value in scores.items():
                    if (name, metric) in expected:
                        group[name, metric] = (value * 100, scores['sample_len'], str(result))
        assert set(group) == expected, (run, set(group), expected)
        assert all(value[1] == counts[key[0]] and 0 <= value[0] <= 100 for key, value in group.items()), run
        assert not data.keys() & group.keys()
        data.update(group)
    return data


def append_replications(wb, seed42):
    """Keep every seed; mean and sample SD require all three completed scores."""
    seeds = (42, 43, 44)
    scores = {(seed, method): (seed42[method] if seed == 42 else collect(method, seed))
              for seed in seeds for method, _ in METHODS[1:]}
    for title, columns in [('目标语言复测', TARGET), ('通用能力复测', GENERAL)]:
        ws = wb.create_sheet(title)
        ws.append(['训练seed', '方法'] + [name for name, _, _ in columns])
        ws.append(['原模型', '原模型'] + [seed42['base'][task, metric][0] for _, task, metric in columns])
        for seed in seeds:
            for method, label in METHODS[1:]:
                row = [seed, label]
                for _, task, metric in columns:
                    value = scores[seed, method].get((task, metric))
                    row.append(value[0] if value else '待完成')
                ws.append(row)
                for col, (_, task, metric) in enumerate(columns, 3):
                    value = scores[seed, method].get((task, metric))
                    if value:
                        ws.cell(ws.max_row, col).comment = Comment('题目/提示数=' + str(value[1]) + '\n' + value[2], '证据')
        ws.append(['说明', '训练seed改变；测评seed固定42；原模型只有一份，不计入训练seed均值。'])
        ws.append(['验收', '每项任务使用全量正式样本且通过评分/逐题/模型身份核验后填值；其他任务待完成。'])
        ws.append(['限制', '固定400步Final；保留全部种子；strict/flexible与ProX格式限制沿用设置页。'])
        for row in range(12, 15):
            ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=10)
        ws.freeze_panes = 'C3'
        for column in 'AB':
            ws.column_dimensions[column].width = 16
        for column in range(3, 11):
            ws.column_dimensions[get_column_letter(column)].width = 20
        ws.print_area = 'A1:J14'
    ws = wb.create_sheet('三种子汇总')
    ws.append(['方法', '任务及主/辅助指标', '完成种子数', 'seed42', 'seed43', 'seed44',
               '三seed均值', '样本标准差', '最小值', '最大值', '原模型', '均值减原模型'])
    for method, label in METHODS[1:]:
        for name, task, metric in TARGET + GENERAL:
            values = [scores[seed, method].get((task, metric)) for seed in seeds]
            row = ws.max_row + 1
            span = f'D{row}:F{row}'
            aggregate = [f'=IF(COUNT({span})=3,{fn}({span}),"待完成")'
                         for fn in ['AVERAGE', 'STDEV', 'MIN', 'MAX']]
            base = seed42['base'][task, metric][0]
            ws.append([label, name, f'=COUNT({span})'] + [v[0] if v else '待完成' for v in values]
                      + aggregate + [base, f'=IF(COUNT({span})=3,G{row}-K{row},"待完成")'])
            for column in [7, 8, 9, 10, 12]:
                ws.cell(row, column).number_format = '0.00'
    ws.append(['说明', 'ACC/EM/通过率为0–100百分数；chrF++为原刻度。逐指标待三seed齐全再计算均值及样本标准差；不对不同任务做混合总分。'])
    ws.merge_cells(start_row=50, start_column=2, end_row=50, end_column=12)
    ws.freeze_panes = 'D2'
    ws.auto_filter.ref = 'A1:L49'
    for column, width in [('A', 17), ('B', 27), ('C', 15)]:
        ws.column_dimensions[column].width = width
    for column in range(4, 13):
        ws.column_dimensions[get_column_letter(column)].width = 16
    # Preserve all selected official metrics, including loose instruction scores.
    included = {(task, metric) for _, task, metric in TARGET + GENERAL}
    evidence = []
    for seed in (43, 44):
        for method, label in METHODS[1:]:
            selected = 0
            for (task, metric), (value, count, path) in sorted(scores[seed, method].items()):
                if (task, metric) in included or task in ['sft_v1_ifeval', 'sft_v1_xifeval_sw']:
                    evidence.append([seed, label, task, metric, count, value, path])
                    selected += 1
            assert 0 <= selected <= 20, (seed, method, selected)
    if evidence:
        ws = wb.create_sheet('复测逐项证据')
        ws.append(['训练seed', '方法', '任务', '官方指标', '题目/提示数', '分数0–100', '原始结果路径'])
        for row in evidence: ws.append(row)
        for column, width in zip('ABCDEFG', [12, 18, 38, 38, 16, 18, 100]):
            ws.column_dimensions[column].width = width
        for row in ws:
            ws.row_dimensions[row[0].row].height = 48 if row[0].row > 1 else 30
            for cell in row:
                cell.alignment = Alignment(vertical='center', wrap_text=True,
                                           horizontal='center' if cell.column in (1, 5, 6) else 'left')
                if cell.column == 6: cell.number_format = '0.00'
        for cell in ws[1]:
            cell.fill = PatternFill('solid', fgColor='183B56')
            cell.font = Font(color='FFFFFF', bold=True)
        ws.freeze_panes = 'D2'
        ws.auto_filter.ref = ws.dimensions
        ws.sheet_view.showGridLines = False
        ws.print_title_rows = '1:1'
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.orientation = 'landscape'
        ws.page_setup.paperSize = ws.PAPERSIZE_A3
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
    for title in ['目标语言复测', '通用能力复测', '三种子汇总']:
        ws = wb[title]
        ws.sheet_view.showGridLines = False
        ws.print_title_rows = '1:1'
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.orientation = 'landscape'
        ws.page_setup.paperSize = ws.PAPERSIZE_A3
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 1 if title != '三种子汇总' else 0
        for row in ws:
            ws.row_dimensions[row[0].row].height = 36
            for cell in row:
                cell.alignment = Alignment(vertical='center', wrap_text=True,
                                           horizontal='center' if cell.column >= 3 else 'left')
                if title == '三种子汇总' and 2 <= cell.row <= 49 and ((cell.row - 2) // 16) % 2:
                    cell.fill = PatternFill('solid', fgColor='F0F6FA')
                if isinstance(cell.value, float):
                    cell.number_format = '0.00'
        for cell in ws[1]:
            cell.fill = PatternFill('solid', fgColor='183B56')
            cell.font = Font(color='FFFFFF', bold=True)
    return scores


def append_ifeval_types(wb):
    source = LOG / 'ifeval-instruction-diagnostics-seed42-20261102.json'
    diagnostic = json.loads(source.read_text())
    assert diagnostic['status'].startswith('Saved official decisions reconstructed')
    ws = wb.create_sheet('指令错误分析')
    ws.append(['官方指令类型（seed42）', '指令数', '原模型正确', 'FFT正确', 'LoRA正确', 'SSU正确',
               'FFT净变化（条）', 'LoRA净变化（条）', 'SSU净变化（条）'])
    types = diagnostic['methods']['full']['types']
    for tag, original in types.items():
        values = [diagnostic['methods'][method]['types'][tag] for method in ['full', 'lora', 'ssu']]
        assert all(v['instructions'] == original['instructions'] and v['base_correct'] == original['base_correct'] for v in values)
        row = ws.max_row + 1
        ws.append([tag, original['instructions'], original['base_correct']] + [v['model_correct'] for v in values]
                  + [f'={column}{row}-C{row}' for column in ['D', 'E', 'F']])
    total = ws.max_row + 1
    ws.append(['合计'] + [f'=SUM({col}2:{col}{total-1})' for col in 'BCDEFGHI'])
    notes = ['本表为首轮seed42；从已保存官方strict逐指令布尔判定统计，四种strict/loose汇总均与原分数一致。',
             '同一提示可含多条指令；这些类型不是独立benchmark，不对类型正确率作不加权平均。净变化可正可负，全部类型保留。',
             '失败包括真实输出未满足引用、项目符号、指定数量等要求，不能一概当作答案抽取错误，也不能全部等同于语义知识丢失。',
             '不运行新推理、不重算另一套评分、不用于选择checkpoint或种子；JSON保留固定首个失败示例。', str(source)]
    for text in notes:
        row = ws.max_row + 1
        ws.append([text])
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=9)
        ws.row_dimensions[row].height = 32
    ws.column_dimensions['A'].width = 56
    ws.column_dimensions['B'].width = 12
    for col in 'CDEF': ws.column_dimensions[col].width = 14
    for col in 'GHI': ws.column_dimensions[col].width = 19
    for row in ws:
        if row[0].row <= total: ws.row_dimensions[row[0].row].height = 26
        for cell in row:
            cell.alignment = Alignment(vertical='center', wrap_text=True, horizontal='left' if cell.column == 1 else 'center')
            if 2 <= cell.row < total and cell.column >= 7: cell.number_format = '+0;-0;0'
    for row in [1, total]:
        for cell in ws[row]:
            cell.fill = PatternFill('solid', fgColor='183B56')
            cell.font = Font(color='FFFFFF', bold=True)
    ws.row_dimensions[1].height = 40
    ws.freeze_panes = 'C2'
    ws.sheet_view.showGridLines = False
    ws.print_title_rows = '1:1'
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = ws.page_setup.fitToHeight = 1


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--include-replications',action='store_true');ap.add_argument('--json-output',type=Path);args=ap.parse_args()
    scores={method:collect(method) for method,_ in METHODS};wb=Workbook();wb.remove(wb.active)
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    navy='183B56';teal='DDEEF0';orange='FFF1D6'
    for title,columns in [('目标语言',TARGET),('英语及通用',GENERAL)]:
        ws=wb.create_sheet(title);n=len(columns)+1
        ws.merge_cells(start_row=1,start_column=1,end_row=1,end_column=n);ws.cell(1,1,'Qwen3-1.7B CL + 多语言基线 | '+title)
        ws.merge_cells(start_row=2,start_column=1,end_row=2,end_column=n);ws.cell(2,1,'快照 '+now+'；Aya SW6400条、一轮400步Final；首轮seed42'+('，独立复测见后页。' if args.include_replications else '。'))
        ws.append(['方法']+[c[0] for c in columns])
        for row,(method,label) in enumerate(METHODS,4):
            ws.cell(row,1,label)
            for col,(_,task,metric) in enumerate(columns,2):
                value=scores[method].get((task,metric));cell=ws.cell(row,col,value[0] if value else '待完成')
                if value:cell.comment=Comment('题目/提示数='+str(value[1])+'\n'+value[2],'证据')
                cell.number_format='0.00';cell.font=Font(color='157044' if value else 'A66B16',size=11)
        ws.merge_cells(start_row=9,start_column=1,end_row=9,end_column=n);ws.cell(9,1,'相对原模型变化：ACC/EM/通过率为百分点；chrF++为分差')
        ws.cell(10,1,'方法')
        for col,(name,_,_) in enumerate(columns,2):ws.cell(10,col,name)
        for row,(_,label) in enumerate(METHODS[1:],11):
            ws.cell(row,1,label)
            for col in range(2,n+1):
                letter=get_column_letter(col);source=row-6
                ws.cell(row,col,f'=IF(ISNUMBER({letter}{source}),{letter}{source}-{letter}$4,"待完成")').number_format='+0.00;-0.00;0.00'
        ws.conditional_formatting.add(f'B11:{get_column_letter(n)}13',ColorScaleRule(start_type='min',start_color='FADBD8',mid_type='num',mid_value=0,mid_color='FFFFFF',end_type='max',end_color='CDEAD8'))
        notes=['† SW ProX原模型受停止序列影响，大量空输出；该列不能解释为纯知识ACC。','‡ flexible为同一生成的辅助提取；strict仍为主指标。数学格式变化与答题能力变化分别解读。','HF阅读/知识与vLLM生成列各沿用自身冻结协议，不把不同列视为同一道题的解析对照。','同一列同题目/提示/解码/评分；完整保留上升、下降和未提升任务；无英语早停门槛。','自有方法：后续阶段，未完成不虚填。当前单seed结果，不声明跨运行BF16逐位重复。']
        for row,text in enumerate(notes,15):
            ws.merge_cells(start_row=row,start_column=1,end_row=row,end_column=n);ws.cell(row,1,text);ws.row_dimensions[row].height=26
        ws.column_dimensions['A'].width=19
        for col in range(2,n+1):ws.column_dimensions[get_column_letter(col)].width=20
        ws.freeze_panes='B4';ws.print_options.horizontalCentered=True;ws.print_area=f'A1:{get_column_letter(n)}19'
        ws.sheet_properties.pageSetUpPr.fitToPage=True;ws.page_setup.orientation='landscape';ws.page_setup.paperSize=ws.PAPERSIZE_A3;ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=1
        for row in [1,3,9,10]:
            for cell in ws[row]:cell.fill=PatternFill('solid',fgColor=navy if row in [1,3,10] else teal);cell.font=Font(color='FFFFFF' if row in [1,3,10] else navy,bold=True,size=12)
        ws.row_dimensions[1].height=32;ws.row_dimensions[2].height=30;ws.row_dimensions[3].height=42;ws.row_dimensions[10].height=42
        for row in ws:
            for cell in row:cell.alignment=Alignment(vertical='center',wrap_text=True,horizontal='left' if cell.column==1 else 'center')
    ws=wb.create_sheet('设置与限制')
    training_seeds = '训练seed42；复测43/44另列；测评seed固定42' if args.include_replications else 'seed42'
    settings=[('项目','Qwen3-1.7B，项目原始后训练模型作为Base；不是另一个Qwen3-1.7B-Base'),('预算','Aya SW6400条；输入2,884,423 token，回答监督1,078,347 token；一轮400步；有效batch16'),('共同设置','非思考；assistant-only；BF16；cutoff2048；不packing；cosine、warmup5%；'+training_seeds),('FFT','全参数；LR1e-5；双卡ZeRO3；micro1 × accum8 × world2'),('LoRA','r16/alpha32/dropout0.05；七种投影；LR1e-4；micro2 × accum8 × world1'),('SSU-Wanda','既有锁定掩码；冻结704,737,792/1,720,574,976≈40.9594%；LR1e-5；WD0；单卡ZeRO2 CPU offload；不是旧SSU15%'),('校准限制','既有Tulu500校准数学占比较高；Aya含机器翻译噪声；不声称完整无污染'),('SW阅读诊断','原模型900题预测B728次、C0次；FFT/LoRA减少此偏置。ACC提升可能含选项校准变化，不能全部视为语义理解提升'),('SW知识诊断','Global-MMLU原模型A/B/C/D预测6245/1970/538/5289次，正确标签分布3222/3462/3582/3776；微调改变偏置，收益来源可能含校准'),('判断规则','固定预算Final为主；不因EN下降提前停止；正式测试不用于调参或选checkpoint'),('HF任务','Belebele EN/SW各900题；MMLU/Global-MMLU各14042题；按各自冻结few-shot/提示/似然评分'),('生成任务','ProX EN/SW各11759题5-shot；GSM8K1319题5-shot；IFEval541提示/834指令0-shot；MGSM SW250题8-shot；SW指令429提示/633指令0-shot'),('翻译','FLORES双向各500对；3-shot；语料级chrF++'),('新训练恢复','首轮seed42三组正式模型均从头训练，未受CPUAdam恢复LR故障影响；SSU checkpoint100与400实际LR/日程一致'),('旧MURI','另行跟踪，不与本表混合；旧Full r2恢复有LR偏差，正确r3续训独立输出'),('源代码与总表','/path/to/project/STATUS.md'),('更新时刻',now)]
    translation_path = LOG / 'translation-generation-seed42-20261102.json'
    translation = json.loads(translation_path.read_text())
    repeat_counts = '/'.join(str(translation['methods'][m]['eng_Latn_to_swh_Latn']['repeated_response_count']) for m, _ in METHODS)
    settings.insert(13, ('翻译输出诊断（seed42）',
        'EN→SW连续重复样本数（每模型500句）：Base/FFT/LoRA/SSU=' + repeat_counts +
        '。规则：忽略大小写、空白分词、1–5词块连续重复至少10次；后验描述。低分含实际生成失败，官方chrF++保留。批次padding宽度不能当作每条截断长度。'))
    ws.append(['设置','说明'])
    for row in settings:ws.append(row)
    ws.column_dimensions['A'].width=23;ws.column_dimensions['B'].width=125
    for row in ws:
        ws.row_dimensions[row[0].row].height=44
        for c in row:c.alignment=Alignment(wrap_text=True,vertical='center')
    for row in ws:
        if row[0].value == '翻译输出诊断（seed42）':
            ws.row_dimensions[row[0].row].height = 72
            row[1].comment = Comment(str(translation_path), '证据')
    ws.freeze_panes='B2';ws.sheet_properties.pageSetUpPr.fitToPage=True;ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=1;ws.page_setup.paperSize=ws.PAPERSIZE_A3
    diagnostic_path=LOG/'math-format-diagnostics-with-prefix-20261102.json'
    diagnostic=json.loads(diagnostic_path.read_text())
    assert diagnostic['status']=='saved official scores unchanged; paired filter categories verified'
    ws=wb.create_sheet('格式诊断')
    ws.append(['方法','数学任务','题数','strict正确','flex正确','strict ACC%','flex ACC%','strict提取失败','其中缺少前缀','提取失败且flex命中','两种均错','原始逐题证据'])
    for method,label in METHODS:
        for task,title in [('sft_v1_gsm8k','EN GSM8K'),('sft_v1_mgsm_native_cot_sw','SW MGSM')]:
            item=diagnostic['methods'][method][task];c=item['counts'];row=ws.max_row+1
            ws.append([label,title,item['samples'],item['strict_correct'],item['flexible_correct'],
                f'=D{row}/C{row}*100',f'=E{row}/C{row}*100',c['strict_invalid'],
                c['strict_invalid_without_expected_prefix'],c['strict_invalid_flexible_correct'],c['both_wrong'],item['samples_file']])
            ws.cell(row,6).number_format='0.00';ws.cell(row,7).number_format='0.00';ws.row_dimensions[row].height=46
    for row,text in enumerate([
      '本页为首轮seed42：仅分析已保存的官方strict/flexible分数与输出，不改变主分数，不运行新推理。完整分类与固定规则示例见JSON证据。',
      'GSM8K strict按官方#### 数字提取，SW MGSM按Jibu ni 数字提取。粗体、货币符号或缺少前缀均可能影响提取。',
      '其中缺少前缀是strict提取失败中的子集；其余失败仍可能由数字格式、加粗等造成，不能一概视为数学算错。',
      'flex命中是辅助数字提取，不等于推理过程有效，也不能替代strict主指标。EN flexible也下降，退化并非全由标记缺失解释。',
      'SW数学strict上升而flex几乎未升，不能据此宣称大幅推理能力提升。',str(diagnostic_path)],12):
        ws.merge_cells(start_row=row,start_column=1,end_row=row,end_column=12);ws.cell(row,1,text);ws.row_dimensions[row].height=34
    for col,width in enumerate([16,18,10,13,13,14,14,18,18,23,14,85],1):ws.column_dimensions[get_column_letter(col)].width=width
    for row in ws:
        for c in row:c.alignment=Alignment(wrap_text=True,vertical='center')
    for c in ws[1]:c.fill=PatternFill('solid',fgColor=navy);c.font=Font(color='FFFFFF',bold=True)
    ws.row_dimensions[1].height=42;ws.freeze_panes='D2';ws.sheet_properties.pageSetUpPr.fitToPage=True;ws.page_setup.orientation='landscape';ws.page_setup.paperSize=ws.PAPERSIZE_A3;ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=1
    ws=wb.create_sheet('训练成本')
    ws.append(['方法','GPU数','启动至退出 分钟','Trainer秒数','Trainer分钟','输入token','监督token','训练阶段输入token/s','GPU小时（Trainer）','来源'])
    for row,(method,label) in enumerate(METHODS[1:],2):
        name='sw_aya6400_'+('ssu_wanda' if method=='ssu' else method)+'_seed42_final_v1'
        folder=Path('/path/to/storage/checkpoints/qwen3_sft_v1_20261101')/name
        run=LOG/name
        start=datetime.fromisoformat((run/'run-start.txt').read_text().splitlines()[0])
        end=datetime.fromisoformat((run/'run-end.txt').read_text().strip())
        runtime=json.loads((folder/'train_results.json').read_text())['train_runtime']
        counts=json.loads((folder/'accounting-rank0.json').read_text())['global_counts']
        ws.append([label,2 if method=='full' else 1,(end-start).total_seconds()/60,runtime,
                   f'=D{row}/60',counts['input_tokens'],counts['shifted_assistant_tokens'],
                   f'=F{row}/D{row}',f'=D{row}/3600*B{row}',str(folder/'train_results.json')])
        for col in [3,4,5,8,9]:ws.cell(row,col).number_format='0.00'
        ws.cell(row,3).comment=Comment(str(run/'run-start.txt')+'\n'+str(run/'run-end.txt'),'证据')
    full_core=json.loads((LOG/'final-full-core-audit.json').read_text())['tasks']
    decoded=sum(round(v['mean_generated_tokens']*v['requests']) for k,v in full_core.items() if k.startswith('backend_'))
    cost_notes=[
      '本页成本为首轮seed42，不含复测及失败尝试。训练均6400条、一轮400步；启动至退出含预检/加载/保存，Trainer含训练及中间保存。',
      '测评与训练分开计时：ProX每语言11759次生成；另有阅读、选择题知识、数学、指令及双向翻译，正式测评可明显长于单次训练。',
      f'FFT核心已完成26057次生成，实际生成{decoded:,} token；逐题输出与停止原因保留。未完成测评不填最终耗时。',
      '同一共享服务器的实测值受CPU offload、磁盘保存、并发及生成长度影响；不按模型参数量单独推断完成时间。',
      '后续提吞吐需要独立短跑验证batch及显存，并对比较模型采用同一冻结设置；当前正式测评继续原协议。']
    for row,text in enumerate(cost_notes,7):
        ws.merge_cells(start_row=row,start_column=1,end_row=row,end_column=10);ws.cell(row,1,text);ws.row_dimensions[row].height=40
    for col,width in enumerate([18,10,22,18,18,19,19,24,23,80],1):ws.column_dimensions[get_column_letter(col)].width=width
    for row in ws:
        for c in row:c.alignment=Alignment(wrap_text=True,vertical='center')
    for c in ws[1]:c.fill=PatternFill('solid',fgColor=navy);c.font=Font(color='FFFFFF',bold=True)
    ws.row_dimensions[1].height=36
    for row in [2,3,4]:ws.row_dimensions[row].height=48
    ws.freeze_panes='C2';ws.sheet_properties.pageSetUpPr.fitToPage=True;ws.page_setup.orientation='landscape';ws.page_setup.paperSize=ws.PAPERSIZE_A3;ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=1
    ws.merge_cells('A14:J14');ws.cell(14,1,'核心正式测评成本：记录启动至结束，含加载/评分与写盘，不含队列等待')
    ws.append(['方法','阶段','题目数','记录墙钟分钟','实际生成token','token/墙钟秒','触及长度上限','原始空输出','官方提取失败','原始计时证据'])
    for method,label in METHODS:
        audit=json.loads((LOG/('base-core-audit.json' if method=='base' else 'final-'+method+'-core-audit.json')).read_text())['tasks']
        for group in ['prox-en','prox-sw','other-core']:
            suffix=group if method=='base' else ('other' if group=='other-core' else group)
            run=LOG/(('base-core-' if method=='base' else 'final-'+method+'-core-')+suffix)
            start=datetime.fromisoformat((run/'run-start.txt').read_text().splitlines()[0]);end=datetime.fromisoformat((run/'run-end.txt').read_text().strip())
            backend=audit['backend_'+group];decoded=round(backend['requests']*backend['mean_generated_tokens'])
            selected=[v for k,v in audit.items() if k.startswith('sft_v1_mmlu_prox_'+group[-2:]+'_')] if group!='other-core' else [audit[k] for k in ['sft_v1_gsm8k','sft_v1_ifeval','sft_v1_mgsm_native_cot_sw','sft_v1_xifeval_sw']]
            empty=sum(next(iter(v['format_diagnostics'].values()))['empty_outputs'] for v in selected)
            failures=sum(v['format_diagnostics']['custom-extract']['answer_extraction_failures'] for v in selected) if group!='other-core' else '见格式诊断'
            row=ws.max_row+1
            ws.append([label,group,backend['requests'],(end-start).total_seconds()/60,decoded,
                f'=E{row}/(D{row}*60)',backend['length_limit'],empty,failures,str(run/'run-start.txt')+' / '+str(run/'run-end.txt')])
            ws.row_dimensions[row].height=40;ws.cell(row,4).number_format='0.00';ws.cell(row,6).number_format='0.00'
    for row,text in [(29,'原生LoRA SW ProX实际用时277.17分钟，输出8,542,529 token；它是本轮评测的重要长尾。'),(30,'这些是共享服务器上的实测成本；同时受生成长度、原生adapter推理和执行环境影响，不据此改变当前科学设置。')]:
        ws.merge_cells(start_row=row,start_column=1,end_row=row,end_column=10);ws.cell(row,1,text);ws.row_dimensions[row].height=40
    for row in [14,15]:
        ws.row_dimensions[row].height=38
        for c in ws[row]:c.fill=PatternFill('solid',fgColor=navy);c.font=Font(color='FFFFFF',bold=True)
    for row in ws:
        for c in row:c.alignment=Alignment(wrap_text=True,vertical='center')
    ws.row_breaks.append(Break(id=13));ws.page_setup.fitToHeight=0
    ws.merge_cells('A33:J33')
    ws.cell(33, 1, '补充正式测评成本 | 原HF/FLORES、首轮seed42；单卡任务阶段墙钟')
    ws.append(['方法', '阶段', '题目/方向数', '墙钟分钟', '原始起止时间证据'])
    ws.merge_cells('E34:J34')
    base_names = {'belebele': 'base17b_belebele_20261031_v1', 'mmlu': 'base17b_mmlu_20261031_r2',
                  'gmmlu': 'base17b_gmmlu_20261031_r2', 'flores': 'base17b_flores_20261031_r2'}
    for method, label in METHODS:
        task_counts = {'belebele': scores[method]['belebele_eng_Latn_locked', 'acc,none'][1] + scores[method]['belebele_swh_Latn_locked', 'acc,none'][1],
                       'mmlu': scores[method]['mmlu_locked', 'acc,none'][1],
                       'gmmlu': scores[method]['global_mmlu_full_sw_locked', 'acc,none'][1],
                       'flores': scores[method]['eng_Latn_to_swh_Latn', 'chrF++'][1] + scores[method]['swh_Latn_to_eng_Latn', 'chrF++'][1]}
        for task in ['belebele', 'mmlu', 'gmmlu', 'flores']:
            run = OLD / base_names[task] if method == 'base' else LOG / f'complementary-{method}-{task}-v1'
            assert (run / 'exit_code.txt').read_text().strip() == '0'
            start_file = run / 'run-start.json'
            if start_file.exists():
                start = datetime.fromisoformat(json.loads(start_file.read_text())['started_at'])
            else:
                start_file = run / 'run-start.txt'
                start = datetime.fromisoformat(start_file.read_text().splitlines()[0])
            end_file = run / 'run-end.txt'
            end = datetime.fromisoformat(end_file.read_text().strip())
            ws.append([label, task, task_counts[task], (end-start).total_seconds()/60, str(start_file) + '\n' + str(end_file)])
            row = ws.max_row
            ws.merge_cells(start_row=row, start_column=5, end_row=row, end_column=10)
            ws.row_dimensions[row].height = 44
            ws.cell(row, 4).number_format = '0.00'
    for row, text in [(52, '各阶段可能与其他GPU任务重叠，不能把阶段时间相加称整体墙钟；本页不含队列等待、复测或失败尝试。'),
                      (53, '选择题每题含多个选项算分，题数不等于前向请求数；翻译1000方向=500对×2。预检、加载、评分和写盘包含在任务阶段墙钟中。')]:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=10)
        ws.cell(row, 1, text)
        ws.row_dimensions[row].height = 40
    for row in ws.iter_rows(min_row=33):
        for cell in row: cell.alignment = Alignment(vertical='center', wrap_text=True, horizontal='center' if 2 <= cell.column <= 4 else 'left')
    for row in [33, 34]:
        ws.row_dimensions[row].height = 38
        for cell in ws[row]:
            cell.fill = PatternFill('solid', fgColor=navy)
            cell.font = Font(color='FFFFFF', bold=True)
    ws.row_breaks.append(Break(id=32))
    ws=wb.create_sheet('逐项证据');ws.append(['方法（首轮seed42）','任务','指标','题目/提示数','分数0–100','原始结果路径'])
    included={(task,metric) for _,task,metric in TARGET+GENERAL}
    for method,label in METHODS:
        for (task,metric),(value,count,path) in sorted(scores[method].items()):
            # Include all four other-core tasks' official loose and strict metrics too.
            if (task,metric) in included or task in ['sft_v1_ifeval','sft_v1_xifeval_sw']:
                ws.append([label,task,metric,count,value,path]);ws.cell(ws.max_row,5).number_format='0.00'
    for key,width in [('A',18),('B',38),('C',35),('D',12),('E',16),('F',100)]:ws.column_dimensions[key].width=width
    
    for row in ws:
        ws.row_dimensions[row[0].row].height=48 if row[0].row>1 else 30
        for cell in row:cell.alignment=Alignment(wrap_text=True,vertical='center',horizontal='center' if cell.column in (4,5) else 'left')
    ws.auto_filter.ref=ws.dimensions;ws.freeze_panes='D2';ws.sheet_properties.pageSetUpPr.fitToPage=True;ws.page_setup.orientation='landscape';ws.page_setup.paperSize=ws.PAPERSIZE_A3;ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=0
    for ws in wb:
        ws.sheet_view.showGridLines=False
        ws.print_options.horizontalCentered=True
        ws.print_title_rows=('1:3' if ws.title in ['目标语言','英语及通用'] else (None if ws.title=='训练成本' else '1:1'))
        if ws.title in ['设置与限制','逐项证据']:
            for c in ws[1]:c.fill=PatternFill('solid',fgColor=navy);c.font=Font(color='FFFFFF',bold=True)
    append_ifeval_types(wb)
    replications = append_replications(wb,scores) if args.include_replications else {}
    args.output.parent.mkdir(parents=True,exist_ok=True);wb.save(args.output)
    check=load_workbook(args.output,data_only=False);assert abs(check['目标语言']['B5'].value-scores['full']['belebele_swh_Latn_locked','acc,none'][0])<1e-10
    assert check['目标语言']['B11'].value=='=IF(ISNUMBER(B5),B5-B$4,"待完成")'
    if args.json_output:
        groups = [(None if method == 'base' else 42, method, data) for method, data in scores.items()]
        groups += [(seed, method, data) for (seed, method), data in replications.items() if seed != 42]
        rows = []
        for seed, method, data in groups:
            for (task, metric), (value, count, path) in sorted(data.items()):
                if (task, metric) not in included and task not in ['sft_v1_ifeval', 'sft_v1_xifeval_sw']:
                    continue
                record = {'training_seed': seed, 'method': method, 'task': task, 'metric': metric,
                          'score_0_to_100': value, 'question_or_prompt_count': count, 'official_result_file': path}
                if metric.startswith('inst_level_'):
                    record['instruction_denominator'] = 834 if task == 'sft_v1_ifeval' else 633
                rows.append(record)
        assert len({(v['training_seed'], v['method'], v['task'], v['metric']) for v in rows}) == len(rows)
        planned = 200 if args.include_replications else 80
        snapshot = {'status': 'Only complete validated formal tasks included; other tasks pending',
                    'snapshot_at': now, 'completed_metric_cells': len(rows), 'planned_metric_cells': planned,
                    'interpretation': 'Metric-cell counts are not wall-time progress; no subset scores, English-retention gate, best-seed choice or mixed-task overall average',
                    'inference_seed': 42, 'checkpoint': 'Fixed400-step Final; shared original Base has training_seed=null',
                    'rows': rows}
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output':str(args.output),'created_at':now,'score_rows':check['逐项证据'].max_row-1,'completed_replication_metric_rows':check['复测逐项证据'].max_row-1 if '复测逐项证据' in check.sheetnames else 0,'sheets':check.sheetnames},ensure_ascii=False))


if __name__=='__main__':main()
