"""Render full and pilot tables without substituting absent results with zero."""
import argparse,json
from pathlib import Path
from common import dump

MODELS=['base','full','lora','ssu']
SPECS=[('MT','flores200_en_sw_full_devtest_v2','EN→SW',['chrF++','BLEU']),('MT','flores200_sw_en_full_devtest_v2','SW→EN',['chrF++','BLEU']),('SUM','xlsum_sw_v2','SW→SW',['ROUGE-1','ROUGE-2','ROUGE-Lsum','chrF++']),('SUM','xlsum_en_v2','EN→EN',['ROUGE-1','ROUGE-2','ROUGE-Lsum','chrF++']),('Code','humaneval_plus_v2','EN→Python',['pass@1_base','pass@1_plus']),('Code','mbpp_plus_v2','EN→Python',['pass@1_base','pass@1_plus'])]

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 data=json.loads(a.results.read_text());index={(r['benchmark'],r['method']):r for r in data['results']};lines=['# Benchmark v2：Full正式结果','', '所有分数按0–100显示。NOT_RUN不等于0，运行中的任务不展示部分分数。','', '| 能力 | Benchmark | 方向 | 指标 | Base | FFT | LoRA | SSU |','|---|---|---|---|---:|---:|---:|---:|']
 for ability,task,direction,metrics in SPECS:
  for metric in metrics:
   values=[]
   for model in MODELS:
    item=index[(task,model)];value=(item.get('metrics') or {}).get(metric)
    values.append(f'{value:.2f}' if value is not None and item['status']=='VERIFIED_COMPLETE' else ('NOT_RUN' if ability=='Code' else item['status']))
   lines.append('| '+' | '.join([ability,task,direction,metric]+values)+' |')
 lines+=['','## 微调相对Base变化（配对95%区间）','','区间未作多重比较校正；只描述同一任务、同一语种、同一组ID的变化。','','| Benchmark | 方法 | 指标 | Δ | 95%区间 |','|---|---|---|---:|---|']
 for c in data['paired_comparisons']:lines.append(f"| {c['benchmark']} | {c['method']} | {c['metric']} | {c['delta']:+.2f} | [{c['ci95_low']:+.2f}, {c['ci95_high']:+.2f}] |")
 state=json.loads(a.state.read_text());pilot=['# 摘要工程Pilot：每语种128篇','','PILOT_ONLY，不是全量正式测试。样本在查看模型评分前冻结。','','| 语言 | 方法 | n | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | chrF++ |','|---|---|---:|---:|---:|---:|---:|']
 for task in ['xlsum_sw_v2','xlsum_en_v2']:
  for model in MODELS:
   s=state['states'].get(f'{model}|{task}|pilot',{'status':'NOT_RUN'})
   if s['status']!='COMPLETE':continue
   summary=json.loads(Path(s['summary']).read_text());scores=summary['metrics'];pilot.append('| '+' | '.join([task,model,str(summary['samples'])]+[f'{scores[x]:.2f}' for x in ['ROUGE-1','ROUGE-2','ROUGE-Lsum','chrF++']])+' |')
 a.output.mkdir(parents=True);(a.output/'paper_table.md').write_text('\n'.join(lines)+'\n');(a.output/'pilot_table.md').write_text('\n'.join(pilot)+'\n');print(a.output)
if __name__=='__main__':main()
