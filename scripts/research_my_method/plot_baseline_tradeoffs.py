"""Frozen single-stage SW Final comparisons; no composite score or selection."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
d=json.loads(a.summary.read_text());lookup={(x['method'],x['task'],x['metric']):x['score_0_to_100'] for x in d['rows']}
methods=[('base','Base','#666666','s'),('full','FFT','#bf584f','o'),('lora','LoRA','#9276b3','o'),('ssu','SSU','#3c8c7b','o'),('sst_norefresh','SST NR','#2866a5','D'),('sst_r10','SST R10','#d28a2f','D')]
configs=[('belebele_swh_Latn_locked','acc,none','sft_v1_ifeval','prompt_level_strict_acc,none','SW reading ACC (%)','EN IFEval strict (%)'),('global_mmlu_full_sw_locked','acc,none','sft_v1_ifeval','prompt_level_strict_acc,none','SW knowledge ACC (%)','EN IFEval strict (%)'),('xlsum_sw_v2','ROUGE-Lsum','sft_v1_gsm8k','exact_match,flexible-extract','SW summary ROUGE-Lsum','EN GSM8K flexible ACC (%)')]
plt.rcParams.update({'pdf.fonttype':42,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(14,5.2));points=[]
for ax,cfg in zip(axes,configs):
 xt,xm,yt,ym,xlabel,ylabel=cfg
 for method,label,color,marker in methods:
  if (method,xt,xm) not in lookup or (method,yt,ym) not in lookup:continue
  x,y=lookup[method,xt,xm],lookup[method,yt,ym];ax.scatter([x],[y],s=55,c=color,marker=marker,zorder=3)
  offset={'Base':(-4,8),'SST NR':(5,-12),'FFT':(-12,-13),'LoRA':(5,6),'SSU':(5,5),'SST R10':(5,6)}[label]
  ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=9,color=color)
  points.append({'method':method,'x_task':xt,'x_metric':xm,'x':x,'y_task':yt,'y_metric':ym,'y':y})
 ax.set(xlabel=xlabel,ylabel=ylabel);ax.margins(x=.25,y=.22);ax.grid(alpha=.15)
fig.suptitle('SW adaptation after 400 updates: learning and retention',fontsize=13)
pending=' R10 pending.' if not any(x['method']=='sst_r10' for x in points) else ''
fig.text(.5,.025,'Qwen3-1.7B; SW6400; one seed. NR: NoRefresh.'+pending+' Recipes differ, including LoRA learning rate.\nHigher is better on every axis; no composite score. Paired uncertainty is reported in accompanying tables.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.94))
for ext in ['png','pdf']:fig.savefig(a.output/('baseline_tradeoffs.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
(a.output/'manifest.json').write_text(json.dumps({'summary_sha256':hashlib.sha256(a.summary.read_bytes()).hexdigest(),'points':points,'causal_or_best_method_claim':False,'files_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in a.output.iterdir()}},indent=2)+'\n')
