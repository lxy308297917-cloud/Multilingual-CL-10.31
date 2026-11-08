"""Separate metrics and languages; display every planned ablation, never rank a winner."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'research/lora_layer_analysis_20261108'
LABELS={'remove_early':'Remove first 12','remove_middle':'Remove middle 12','remove_late':'Remove last 12','keep_late':'Keep last 12','remove_attention':'Remove Attention','remove_mlp':'Remove MLP'}
def main():
 rows=list(csv.DictReader((O/'ablation_scores.csv').open()));variants=list(LABELS);metrics=[('Belebele target','ACC'),('Belebele EN','ACC'),('XL-Sum target','ROUGE-Lsum'),('GSM8K','flexible-extract'),('IFEval','prompt_level_strict_acc')]
 fig,axes=plt.subplots(1,3,figsize=(18,5))
 for ax,lang in zip(axes,['sw','ar','ru']):
  subset={(x['variant'],x['task'],x['metric']):x for x in rows if x['language']==lang};ax.axhline(0,color='gray',linewidth=0.8);ax.axvline(0,color='gray',linewidth=0.8)
  for i,v in enumerate(variants):
   if (v,*metrics[0]) not in subset or (v,*metrics[4]) not in subset:continue
   x=subset[(v,*metrics[0])];y=subset[(v,*metrics[4])];a=float(x['delta_vs_full']);b=float(y['delta_vs_full']);ax.errorbar(a,b,xerr=[[max(0,a-float(x['ci95_low']))],[max(0,float(x['ci95_high'])-a)]],yerr=[[max(0,b-float(y['ci95_low']))],[max(0,float(y['ci95_high'])-b)]],fmt='o',color=plt.cm.tab10(i),capsize=2,label=LABELS[v]);ax.annotate(str(i+1),(a,b),xytext=(4,4),textcoords='offset points',fontsize=8)
  ax.set_title(lang.upper());ax.set_xlabel('Target Belebele change vs Full (pp)');ax.set_ylabel('IFEval prompt strict change vs Full (pp)')
 axes[-1].legend(loc='best',fontsize=7);fig.suptitle('Post-training ablations: target reading vs instruction following (single seed42)');fig.tight_layout();fig.savefig(O/'reading_ifeval_tradeoffs.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,5,figsize=(19,8));labels=[l.upper()+' / '+LABELS[v] for l in ['sw','ar','ru'] for v in variants]
 for ax,metric in zip(axes,metrics):
  lookup={(x['language'],x['variant']):float(x['delta_vs_full']) for x in rows if (x['task'],x['metric'])==metric};values=np.array([lookup.get((l,v),np.nan) for l in ['sw','ar','ru'] for v in variants]);scale=max(1,float(np.nanmax(abs(values)))) if np.isfinite(values).any() else 1;image=ax.imshow(values.reshape(-1,1),cmap='RdBu_r',vmin=-scale,vmax=scale,aspect='auto');ax.set_xticks([]);ax.set_yticks(range(18),labels,fontsize=7 if ax==axes[0] else 0);ax.set_title(metric[0]+'\n'+('ROUGE-Lsum' if metric[0]=='XL-Sum target' else 'percentage points'),fontsize=9)
  if ax!=axes[0]:ax.set_yticklabels([])
  for j,value in enumerate(values):
   if np.isfinite(value):ax.text(0,j,f'{value:+.2f}',ha='center',va='center',fontsize=8)
  fig.colorbar(image,ax=ax,fraction=.05)
 fig.suptitle('All ablations vs Full: separate color scale for each metric, no aggregate score');fig.tight_layout();fig.savefig(O/'all_metric_changes.png',dpi=180);plt.close(fig)
 print('Plots generated from',len(rows),'complete metric rows')
if __name__=='__main__':main()
