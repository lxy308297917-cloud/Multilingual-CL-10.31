"""Post-hoc score interaction using four already planned masks, no extra inference.

I(E,M | L=on) = Full - RemoveEarly - RemoveMiddle + KeepLate.
This is a finite difference of benchmark scores, not proof of gradient conflict.
"""
import importlib.util,json,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];S=R/'scripts/lora_layer_analysis';spec=importlib.util.spec_from_file_location('analysis_helpers',S/'collect.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
 assert m.load(m.O/'evaluation_verification.json')['complete'],'Run only after all primary cases complete'
 rng=np.random.default_rng(42);rows=[]
 def add(lang,task,metric,values,counts=None):
  full,eoff,moff,late=[np.asarray(v,dtype=float) for v in values];assert full.shape==eoff.shape==moff.shape==late.shape;d=full-eoff-moff+late;w=np.ones(len(d)) if counts is None else np.asarray(counts);delta=100*d.sum()/w.sum();boot=[]
  for _ in range(2000):idx=rng.integers(0,len(d),len(d));boot.append(100*d[idx].sum()/w[idx].sum())
  lo,hi=np.percentile(boot,[2.5,97.5]);rows.append(dict(language=lang,task=task,metric=metric,interaction=delta,ci95_low=float(lo),ci95_high=float(hi),bootstrap_unit='prompt_cluster' if counts is not None else 'item',scope='POST_HOC_EXPLORATORY_NOT_USED_FOR_MODEL_SELECTION'))
 for lang in ['sw','ar','ru']:
  base=m.L/'formal'/lang;models=['Full','remove_early','remove_middle','keep_late']
  hf_full=m.SW/'lora-harness-v1/belebele/results' if lang=='sw' else m.G/(lang+'-lora-pipeline-v1/hf/results');core_full=m.SW/'lora-generation-v2/core/results' if lang=='sw' else m.G/('recovery-20261108-v1/'+lang+'-core/results');summary_full=m.SW/'lora-generation-v2/xlsum_sw_v2-scores' if lang=='sw' else (m.G/'recovery-20261108-v1/ru-summary/xlsum_ru_v1-scores' if lang=='ru' else m.G/'ar-lora-pipeline-v1/xlsum_ar_v1-scores')
  for target in ['target','EN']:
   task=('belebele_swh_Latn_locked' if target=='target' else 'belebele_eng_Latn_locked') if lang=='sw' else 'aya4bml_belebele_'+(lang if target=='target' else 'en');full=m.samples(hf_full,task);values=[]
   for label in models:
    source=full if label=='Full' else m.samples(base/label/'hf/results',task);a,b=m.pair(full,source);assert len(b)==900;values.append([x['acc'] for x in b])
   add(lang,'Belebele '+target,'ACC',values)
  for task in ['sft_v1_gsm8k','sft_v1_ifeval']:
   full=m.samples(core_full,task);by_model=[]
   for label in models:
    source=full if label=='Full' else m.samples(base/label/'core/results',task);a,b=m.pair(full,source);by_model.append(b)
   if task.endswith('gsm8k'):
    for filt in ['strict-match','flexible-extract']:add(lang,'GSM8K',filt,[[x['exact_match'] for x in b if x['filter']==filt] for b in by_model])
   else:
    add(lang,'IFEval','prompt_level_strict_acc',[[x['prompt_level_strict_acc'] for x in b] for b in by_model]);counts=[len(x['inst_level_strict_acc']) for x in by_model[0]];add(lang,'IFEval','inst_level_strict_acc',[[sum(x['inst_level_strict_acc']) for x in b] for b in by_model],counts)
  values=[];keys=None
  for label in models:
   root=summary_full if label=='Full' else base/label/'summary-scores'
   data=m.load(root/'rouge_per_sample.json') if lang=='sw' else m.lines(root/'per_sample_scores.jsonl');lookup={x['id']:x for x in data}
   if keys is None:keys=sorted(lookup)
   assert keys==sorted(lookup);score='ROUGE-Lsum' if lang=='sw' else 'rougeLsum';values.append([lookup[k][score]/100 for k in keys])
  add(lang,'XL-Sum target','ROUGE-Lsum',values)
 with (m.O/'exploratory_interactions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 text=['# 探索性：Early/Middle在Late启用条件下的分数交互','', '此分析在观察SW首批阅读及核心结果后提出，使用原预设四个掩码，不新增推理。不是预注册主结果，不用于选择B的层数或新方法。','', '定义：I = S(Full) − S(Remove-Early) − S(Remove-Middle) + S(Keep-Late)。它只描述这个训练后Adapter和指标上的非加性；ACC阈值、指标非线性及层间不匹配都可能产生交互，不能据此宣称梯度冲突或神经元功能定位。','', '|语言|任务|交互|配对95%区间|','|---|---|---:|---|']
 for x in rows:text.append(f"|{x['language']}|{x['task']} {x['metric']}|{x['interaction']:+.2f}|[{x['ci95_low']:+.2f},{x['ci95_high']:+.2f}]|")
 text+=['','除摘要为ROUGE分外，交互单位均为百分点。单seed42，2000次同题配对bootstrap；指令级按prompt cluster。未做多重比较校正，不外推为跨种子稳定或跨模型机制。Global-MMLU不在此额外探索范围，原完整分数仍保留在主结果表。']
 (m.O/'exploratory_interactions.md').write_text('\n'.join(text)+'\n');print('Exploratory interaction rows',len(rows))
if __name__=='__main__':main()
