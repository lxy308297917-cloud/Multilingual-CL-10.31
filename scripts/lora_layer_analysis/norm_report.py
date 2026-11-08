"""Describe layerwise update magnitude without claiming functional attribution."""
from pathlib import Path
import csv,json,itertools
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'research/lora_layer_analysis_20261108'
def main():
 rows=list(csv.DictReader((O/'module_norms.csv').open()));audit=json.loads((O/'adapter_ablation_audit.json').read_text());curves={};parts=[]
 for lang in ['sw','ar','ru']:
  xs=[x for x in rows if x['language']==lang];norm2=sum(float(x['delta_frobenius'])**2 for x in xs);curves[lang]=np.array([sum(float(x['delta_frobenius'])**2 for x in xs if int(x['layer'])==l) for l in range(36)])/norm2
  for group in ['early','middle','late','attention','mlp']:
   sub=[x for x in xs if x['group']==group or x['kind']==group];parts.append(dict(language=lang,group=group,modules=len(sub),adapter_parameters=sum(int(x['adapter_parameters']) for x in sub),joint_delta_frobenius=sum(float(x['delta_frobenius'])**2 for x in sub)**0.5,delta_squared_norm_share=sum(float(x['delta_frobenius'])**2 for x in sub)/norm2))
 with (O/'group_norms.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(parts[0]));w.writeheader();w.writerows(parts)
 comparisons=[dict(language_a=a,language_b=b,layer_squared_norm_profile_cosine=float(np.dot(curves[a],curves[b])/(np.linalg.norm(curves[a])*np.linalg.norm(curves[b])))) for a,b in itertools.combinations(curves,2)]
 (O/'norm_profile_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
 fig,ax=plt.subplots(figsize=(8,4))
 for lang,v in curves.items():ax.plot(range(36),v,label=lang)
 ax.axvline(11.5,color='gray',linestyle='--');ax.axvline(23.5,color='gray',linestyle='--');ax.set_xlabel('Layer (0-based)');ax.set_ylabel('Share of squared effective update norm');ax.legend();fig.tight_layout();fig.savefig(O/'three_language_norm_profiles.png',dpi=180);plt.close(fig)
 s=['# A1：有效LoRA更新范数','', '使用alpha/r=2及FP64因子Gram恒等式计算||BA||F，已与显式矩阵乘积小例对照。相对更新以同一Base各矩阵Frobenius norm归一化。有效更新是已训练参数差值，不是梯度。','', '|语言|分组|模块数|Adapter参数|联合范数|平方范数占比|','|---|---|---:|---:|---:|---:|']
 for x in parts:s.append(f"|{x['language']}|{x['group']}|{x['modules']}|{x['adapter_parameters']}|{x['joint_delta_frobenius']:.4f}|{x['delta_squared_norm_share']*100:.2f}%|")
 s+=['','## 消融范围','', '|变体|启用模块|保留Adapter参数|关闭Adapter参数|','|---|---:|---:|---:|']
 for x in audit['variants'][:6]:s.append(f"|{x['variant']}|{x['active_modules']}|{x['active_adapter_parameters']}|{x['disabled_adapter_parameters']}|")
 s+=['','36层每12层分组；原始每语种252模块/33,030,144个Adapter参数。表中保留参数表示推理增量的范围，不是新的训练参数预算。Attention与MLP的矩阵大小、保留参数数目不同，比较时不能忽略。','', '范数分布相似不等于功能分布相似；任何层级保护或遗忘因果解释必须结合A2/A3实际任务结果，当前范数本身不支持选择Last-k。','', '![三语言范数分布](three_language_norm_profiles.png)']
 (O/'norm_report.md').write_text('\n'.join(s)+'\n');print('Norm report complete')
if __name__=='__main__':main()
