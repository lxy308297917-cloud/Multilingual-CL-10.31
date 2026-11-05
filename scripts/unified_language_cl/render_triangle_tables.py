"""Render literal ES/TH stage scores from verified fixed-Final matrices."""
import csv,json
from pathlib import Path
from common import ROOT,sha,dump
TRACKS=['massive','xnli','mtop','xlsum','wikilingua','sib200','mathoctopus']
TITLES={'massive':'MASSIVE','xnli':'XNLI','mtop':'MTOP','xlsum':'XL-Sum','wikilingua':'WikiLingua','sib200':'SIB-200','mgsm_rev2':'MathOctopus / MGSM-Rev2','msvamp':'MathOctopus / MSVAMP'}
def main():
 records=[];sources={}
 for track in TRACKS:
  path=ROOT/'results/unified_language_cl'/track/'comparison-test-v1/effects.json';v=json.loads(path.read_text());assert v['status']=='COMPLETE_FIXED_FINAL_MATRIX' and v['same_prompts_references_ids_verified']
  assert v['model_order']==['base','single_es','single_th','seq_es_th'];sources[track]=sha(path)
  for bench in (['mgsm_rev2','msvamp'] if track=='mathoctopus' else [track]):
   es=v['groups']['es:'+bench];th=v['groups']['th:'+bench]
   unit='ROUGE-Lsum' if track in ['xlsum','wikilingua'] else 'Tree EM%' if track=='mtop' else 'ACC%'
   r={'track':track,'benchmark':bench,'label':TITLES[bench],'unit':unit,'es_n':es['samples'],'th_n':th['samples'],'es_base':es['scores']['base'],'th_base':th['scores']['base'],'es_stage1':es['scores']['single_es'],'th_stage1_measured':th['scores']['single_es'],'es_stage2':es['scores']['seq_es_th'],'th_stage2':th['scores']['seq_es_th'],'th_direct':th['scores']['single_th']};records.append(r)
 lines=['# ES→TH 三角形阶段结果表','', 'Task 1 = 西语（ES），Task 2 = 泰语（TH）。Stage 1 为 Base→ES Final；Stage 2 为共享 ES Final→TH Final。下面展示实际分数，不是分数变化。','', '原生任务同时测了 ES/TH/FR/EN；本表只取两个训练语言。`—` 是按三角形展示省略未训练语言，TH 在 Stage 1 的实测成绩仍保留在机器可读表和[完整四语言矩阵](stage_language_matrix.md)。GSM8K、IFEval及EN Belebele是另列的英语通用监测，不是Task 1/2的泰语版本。','', '| Benchmark／指标 | 训练阶段 | Task 1：西语 ES | Task 2：泰语 TH |','|---|---|---:|---:|']
 for r in records:
  lines.append(f"| **{r['label']} / {r['unit']}** | Stage 1：ES | {r['es_stage1']:.2f} | — |")
  lines.append(f"| | Stage 2：ES→TH | {r['es_stage2']:.2f} | {r['th_stage2']:.2f} |")
 lines+=['','## 判断目标学习所需的基线','', '三角表中的Task 1上下比较表示旧任务保持；Task 2的Final还应与原模型和直接泰语训练比较。SEQ累计两阶段预算，直接TH只有一阶段，不能当等总算力比较。','', '| Benchmark | ES原模型 | TH原模型 | 直接TH Final | ES→TH的TH Final |','|---|---:|---:|---:|---:|']
 for r in records:lines.append(f"| {r['label']} | {r['es_base']:.2f} | {r['th_base']:.2f} | {r['th_direct']:.2f} | {r['th_stage2']:.2f} |")
 lines+=['','单位：分类和数字答案为ACC%，MTOP为结构树精确匹配%，摘要为ROUGE-Lsum F1分，不跨任务平均。MathOctopus两个数学任务Final均低于原模型。','', 'WikiLingua为官方划分的唯一映射非空子集，非全量官方test。模型均固定seed42与预定Final，没有依据这些分数早停或选择checkpoint。变化的配对区间见[FFT-SEQ结果](fft_seq_results.md)。', '', '[机器可读实际分数](../../results/unified_language_cl/triangle-tables-v1/scores.csv)保留未舍入值、样本量和Stage 1泰语实测值；同一分数的不同表格仅改变展示形式。']
 out=ROOT/'results/unified_language_cl/triangle-tables-v1';assert not out.exists();out.mkdir()
 with (out/'scores.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator="\n");w.writeheader();w.writerows(records)
 dump(out/'verification.json',{'status':'TRIANGLE_TABLES_FROM_ALL_SEVEN_VERIFIED_MAIN_MATRICES','source_effects_sha256':sources,'rows':len(records),'no_new_scoring':True,'no_new_inference':True,'omitted_cells':'TH Stage1 measured but omitted only for triangular display','scores_csv_sha256':sha(out/'scores.csv')})
 doc=ROOT/'research/unified_language_cl/triangle_tables.md';assert not doc.exists();doc.write_text('\n'.join(lines)+'\n');print(json.dumps({'status':'COMPLETE','benchmarks':len(records),'source_matrices':len(sources)}))
if __name__=='__main__':main()
