"""Balanced result tables from completed aggregate files; no model selection."""
import argparse,json
from pathlib import Path

PRIMARY = [
 ('HA reading ACC','HA Belebele|acc|none',100),
 ('HA knowledge ACC','HA Global-MMLU|acc|none',100),
 ('SW reading ACC','SW Belebele|acc|none',100),
 ('SW knowledge ACC','SW Global-MMLU|acc|none',100),
 ('EN reading ACC','EN Belebele|acc|none',100),
 ('EN knowledge ACC','EN MMLU|acc|none',100),
 ('GSM8K strict','sft_v1_gsm8k|exact_match|strict-match',100),
 ('GSM8K flexible','sft_v1_gsm8k|exact_match|flexible-extract',100),
 ('IFEval prompt strict','sft_v1_ifeval|prompt_level_strict_acc|none',100),
 ('IFEval instruction strict','sft_v1_ifeval|inst_level_strict_acc|none',100),
 ('SW MGSM flexible','sft_v1_mgsm_native_cot_sw|exact_match|flexible-extract',100),
 ('SW X-IFEval prompt strict','sft_v1_xifeval_sw|prompt_level_strict_acc|none',100),
 ('FLORES EN to SW chrF++','en_sw|chrF++',1),
 ('FLORES SW to EN chrF++','sw_en|chrF++',1)]
READOUTS=['raw_letter','rotation_content_average','content_mean','content_raw','content_rotation_average','crossed16_content_average']

def main():
 p=argparse.ArgumentParser()
 for key in ['formal','fresh','cost','output']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();f=json.loads(a.formal.read_text());h=json.loads(a.fresh.read_text());cost=json.loads(a.cost.read_text())
 assert not a.output.exists()
 def ci(v,scale):return f"{scale*v['delta']:+.2f} [{scale*v['low']:+.2f}, {scale*v['high']:+.2f}]"
 lines=['# 固定100步：主指标与配对比较','','W1为SW FFT400起点；A=HA-SFT，B=标签边际，C=内容决策KD，D=同题答案token回放。全部HA阶段固定100步/1600样本，沿用400步scheduler。正式测试有历史曝光，只作固定配方的补充比较。完整原生指标见formal_table.md；以下任务集合按既定能力类别同时报告正负变化。','',
 '| Benchmark / metric | n | W1 | A SFT | B marginal | C content KD | D replay |','|---|---:|---:|---:|---:|---:|---:|']
 for title,key,scale in PRIMARY:
  n=f['scores']['W1'][key]['n'];values=[f"{f['scores'][m][key]['score']*scale:.2f}" for m in ['W1','A','B','C','D']]
  lines.append('| '+' | '.join([title,str(n)]+values)+' |')
 lines+=['','## 内容KD与各对照的配对差值','','ACC为百分点，chrF++为原始分值差。每格点估计[95%区间]；固定模型下对原题/段落/提示bootstrap，未多重校正，不是训练种子区间。','',
 '| Metric | C minus A | C minus B | C minus D |','|---|---:|---:|---:|']
 for title,key,scale in PRIMARY:
  lines.append('| '+' | '.join([title]+[ci(f['contrasts']['C_minus_'+m][key],scale) for m in ['A','B','D']])+' |')
 lines+=['','## Hausa学习必须比较阶段2起点','','不是比较最终绝对分数与原始W0，而是各100步模型减去共同W1。','',
 '| Target task | A minus W1 | B minus W1 | C minus W1 | D minus W1 |','|---|---:|---:|---:|---:|']
 for title,key,scale in PRIMARY[:2]:
  lines.append('| '+' | '.join([title]+[ci(f['contrasts'][m+'_minus_W1'][key],scale) for m in ['A','B','C','D']])+' |')
 lines+=['','## 新平行留出题：全部固定读出','','每语言120个原题ID，三个语言共同分组。不是官方阅读/知识benchmark的替代指标；完整内容token均分、原始LP、轮换平均及交叉标签控制均保留。','',
 '| Model | Language | n | Letter | Rotation | Content mean | Content raw | Content rotation | Crossed16 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for model in ['W1','A','B','C','D']:
  for lang in ['en','sw','ha']:
   metrics=h['summaries'][model][lang]['metrics']
   lines.append('| '+' | '.join([model,lang,str(metrics['raw_letter']['n'])]+[f"{metrics[k]['mean']*100:.2f}" for k in READOUTS])+' |')
 lines+=['','## 留出题C与两个额外数据/计算对照的差值','','全部为百分点[95%语义ID配对区间]，不选择最有利读出。','',
 '| Language / readout | C minus B | C minus D |','|---|---:|---:|']
 for lang in ['en','sw','ha']:
  for key in READOUTS:
   lines.append('| '+' | '.join([lang+' / '+key]+[ci(dict(delta=h['contrasts']['C_minus_'+m][lang][key]['mean'],low=h['contrasts']['C_minus_'+m][lang][key]['low'],high=h['contrasts']['C_minus_'+m][lang][key]['high']),100) for m in ['B','D']])+' |')
 lines+=['','## 实际新增训练与数据曝光','','A复用已验证旧100检查点，当前新增训练为0。每组HA1600条、653205输入token、231035 assistant监督token；B/C/D共同50个旧任务语义ID、EN/SW各50次、四轮提示。','',
 '| Arm | Main training wall min | Two-GPU wall proxy hours | HA examples | Old logical input tokens | Old physical input tokens |','|---|---:|---:|---:|---:|---:|']
 for m in ['B','C','D']:
  v=cost['training'][m];lines.append('| '+' | '.join([m,f"{v['training_process_wall_seconds']/60:.2f}",f"{v['gpu_wall_hours_proxy']:.3f}",str(v['main_exposure']['samples']),str(v['aux_logical_input_tokens']),str(v['aux_physical_input_tokens'])])+' |')
 lines+=['',f"离线教师物理上计算一次：{cost['teacher']['physical_cost']['seconds']:.2f}秒，{cost['teacher']['physical_cost']['gpu_wall_hours']:.4f} GPU墙钟小时。B/C/D使用相同教师预算，物理共享成本只记一次；A没有附加教师/旧任务计算，不称四组总计算相同。",'',
 f"本轮总新增GPU墙钟代理 {cost['total']['all_new_gpu_wall_hours_proxy']:.3f}小时；包含三组训练、两个工程探针、教师、正式评测与新读出。它含模型加载/核验/保存CPU阶段，不是精确kernel时间或FLOPs。"]
 a.output.write_text('\n'.join(lines)+'\n');print('Wrote complete balanced comparison tables')
if __name__=='__main__':main()
