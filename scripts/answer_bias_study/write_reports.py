from pathlib import Path
import argparse, csv, datetime, json

LABELS={'base':'Base','sw_fft':'SW FFT','sw_ha_fft':'SW→HA FFT','ha_only':'HA-only FFT','sw_sst_norefresh':'SW SST NoRefresh','sw_sst_r10':'SW SST R10','sw_ha_sst_norefresh':'SW→HA SST NoRefresh'}
METHODS={'raw_letter':'原字母','position_average':'位置平均','content_mean':'内容均分','content_raw':'内容总分','content_position_average':'内容位置平均','pride':'PriDe','blank_calibration':'空输入校准'}


def f(v):return f'{v*100:.2f}'
def interval(v):return f"{f(v['mean'])} [{f(v['low'])}, {f(v['high'])}]"


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.analysis.read_text());a.output.mkdir(exist_ok=True);summ=d['summaries']
    lines=['# 固定模型的答案偏好与内容评分结果','',f"更新：{datetime.datetime.now().astimezone().isoformat()}。七个模型全部完成，无新增训练。",'',
           '所有分数是本轮冻结诊断协议的百分数，**不是新的全量正式benchmark**。知识每语言219题0-shot；阅读每语言179题，历史已曝光。校准材料为另53个官方知识dev语义ID。原正式900/14042分数不覆盖。', '',
           '## Hausa主要读出对照','', '| 模型 | 阅读原字母 | 阅读位置平均 | 阅读内容均分 | 知识原字母 | 知识位置平均 | 知识内容均分 |','|---|---:|---:|---:|---:|---:|---:|']
    for model,groups in summ.items():
        vals=[groups[t+'|ha']['metrics'][m]['mean'] for t in ['reading_exposed','knowledge_dev'] for m in ['raw_letter','position_average','content_mean']]
        lines.append('| '+LABELS[model]+' | '+' | '.join(f(v) for v in vals)+' |')
    lines+=['','## Hausa位置依赖','', '| 模型 | 任务 | 四轮A选择占比 | 字母映射后内容全一致 | 无字母内容全一致 | 平均后近并列题数 |','|---|---|---:|---:|---:|---:|']
    for model,groups in summ.items():
        for task in ['reading_exposed','knowledge_dev']:
            v=groups[task+'|ha'];lines.append(f"| {LABELS[model]} | {task} | {f(v['label_share_all_rotations'][0])} | {f(v['letter_unanimity'])} | {f(v['content_unanimity'])} | {v['position_average_probability_gap_below1e6']}/{v['n']} |")
    lines+=['','四轮平均中的金答案位置完全平衡，A占比不再受原金答案分布影响。但A字母与第一展示位置仍绑定，不能单独区分token偏好与顺序偏好。近并列阈值为平均候选概率差≤1e-6；精确容差1e-12的并列信用ACC另存机器结果。','','## 全模型、全语言ACC与95%区间','', '区间以固定模型为条件，阅读按段落、知识按题配对抽样2000次；未多重校正。', '', '| 模型 | 任务/语言 | 原字母 | 位置平均 | 内容均分 | 内容总分 | 内容位置平均 |','|---|---|---:|---:|---:|---:|---:|']
    csvrows=[]
    for model,groups in summ.items():
        for g,v in groups.items():
            methods=['raw_letter','position_average','content_mean','content_raw','content_position_average']
            lines.append('| '+LABELS[model]+' | '+g+' | '+' | '.join(interval(v['metrics'][m]) for m in methods)+' |')
            for m,score in v['metrics'].items():csvrows.append(dict(model=model,group=g,method=m,**score))
    lines+=['','## 固定校准及随机对照','', '仅对达到冻结位置偏好门槛的模型/任务/语言执行校准比较。知识先验来自独立53题；阅读上的PriDe是知识→阅读迁移，不是同分布拟合。未通过门槛的组合为NOT_RUN，不用其分数挑选是否校准。随机对照为同一先验的全部23种非恒等映射，报告均值而非最优值。', '', '| 模型 | 任务/语言 | PriDe | 空输入校准 | 随机先验均值 | PriDe−随机均值 |','|---|---|---:|---:|---:|---:|']
    for model,groups in summ.items():
        for g,v in groups.items():
            if v['calibration_gate']:
                random=v['random_prior_control'];lines.append('| '+LABELS[model]+' | '+g+' | '+interval(v['metrics']['pride'])+' | '+interval(v['metrics']['blank_calibration'])+' | '+interval(random['mean'])+' | '+interval(random['pride_minus_random_mean'])+' |')
            else:lines.append('| '+LABELS[model]+' | '+g+' | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |')
    lines+=['','## 顺序适配的同题变化','', '下表为Final减自身阶段起点，单位pp。交互列为“位置平均的变化−原字母的变化”；改变测量方式不等于恢复训练。', '', '| 阶段 | 任务/语言 | 原字母变化 | 位置平均变化 | 内容均分变化 | 读出交互 |','|---|---|---:|---:|---:|---:|']
    for name,groups in d['contrasts'].items():
        for g,v in groups.items():lines.append('| '+name+' | '+g+' | '+' | '.join(interval(v[m]) for m in ['raw_letter','position_average','content_mean','position_average_change_minus_raw_change'])+' |')
    lines+=['','## 计算与验真','', f"七模型合计占用墙钟代理 **{d['total_gpu_wall_hours']:.3f} GPU小时**（包括加载和保存，不是CUDA kernel时间）。没有新增训练token或optimizer步。", '', '| 模型 | 秒 | 题目×语言条数（含校准） | 选项分数 | 峰值显存GiB |','|---|---:|---:|---:|---:|']
    for model,v in d['compute'].items():lines.append(f"| {LABELS[model]} | {v['seconds']:.1f} | {v['items']} | {v['option_scores']} | {v['peak_cuda_allocated']/1024**3:.2f} |")
    lines+=['','CPU6项检查通过；每个真实模型在校准题上核对标准完整前向与答案位置评分。完整数据/权重/源码/启动日志指纹及逐题预测留在本地。公开版不含主机路径、进程/设备清单或原始数据。首次shell后台协调器未启动Python、无模型产物，改用独立会话启动同一计划；没有重跑科学模型。','','24全排列、权重干预、新FFT/SST训练、独立双语语义认证：NOT_RUN。校准与内容评分的改善不证明内部知识完整，也不构成My Method。', '']
    (a.output/'results.md').write_text('\n'.join(lines))
    with (a.output/'paper_table.csv').open('w') as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    lines=['# 跨语言同题决策分析','', '每个语义ID在EN/SW/HA共同纳入，选项内容ID与金答案已对齐。阅读采用官方专业翻译；知识Full的翻译语义未经独立认证，分析仅探索性。不是不同语言非平行任务的直接作差。','','编码顺序为EN/SW/HA：100表示仅EN正确，110表示EN和SW正确而HA错误，111三语言全对，000全错。选项排列和ID固定；学科和Base EN难度分层见机器结果。','','| 模型 | 任务 | 读出 | N | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 | 三语言同内容决策% |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model,tasks in d['cross_language'].items():
        for task,methods in tasks.items():
            for method,v in methods.items():
                pats=v['correctness_patterns'];lines.append('| '+LABELS[model]+' | '+task+' | '+METHODS[method]+f" | {v['n']} | "+' | '.join(str(pats.get(k,0)) for k in ['000','001','010','011','100','101','110','111'])+f" | {f(v['all_language_content_id_agreement'])} |")
    lines+=['','三语言选同一个内容既可能共同答对，也可能共同答错；一致率必须与111和000一起解释。去标签内容续写会改变输入要求，不能把与字母形式的差别全当成“语义能力”。位置平均后仍有退化，更不能仅凭原始A偏好解释全部能力变化。','','逐题变对/变错、内容决策改变率、全部学科结果与难度组结果保留于机器分析；不据其中高分子组重新筛选方法或题目。', '']
    (a.output/'cross_language_analysis.md').write_text('\n'.join(lines))
    print(dict(reports=['results.md','cross_language_analysis.md'],rows=len(csvrows)))

if __name__=='__main__':main()
