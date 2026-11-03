from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
d=json.loads(a.analysis.read_text());models=list(d['summaries']);labels=['Base','SW FFT','SW→HA FFT','HA-only','SST SW','SST R10 SW','SST SW→HA']
fig,axes=plt.subplots(2,3,figsize=(18,9),sharey=True)
methods=[('raw_letter','Letter', '#64748b'),('position_average','Position average','#2563eb'),('content_mean','Content/token','#d97706'),('pride','Independent prior','#16a34a')]
for row,task in enumerate(['reading_exposed','knowledge_dev']):
 for col,lang in enumerate(['en','sw','ha']):
  ax=axes[row,col]
  for k,(m,name,color) in enumerate(methods):
   for j,model in enumerate(models):
    v=d['summaries'][model][task+'|'+lang]['metrics'].get(m)
    if not v:continue
    ax.errorbar(j+(k-1.5)*.14,100*v['mean'],yerr=[[100*(v['mean']-v['low'])],[100*(v['high']-v['mean'])]],fmt='o',ms=4,color=color,capsize=2,label=name if j==0 else None)
  ax.set_xticks(np.arange(len(models)));ax.set_xticklabels(labels,rotation=35,ha='right',fontsize=9)
  ax.axhline(25,color='#cbd5e1',ls='--',lw=1);ax.grid(axis='y',alpha=.2)
  ax.set_title(('Reading (179 exposed)' if row==0 else 'Knowledge (219 dev)')+' · '+lang.upper());ax.set_ylim(0,100)
  if col==0:ax.set_ylabel('Accuracy (%) · paired 95% intervals')
handles=[plt.Line2D([],[],color=c,marker='o',ls='',label=n) for _,n,c in methods]
fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,.97))
fig.suptitle('Frozen diagnostic cohorts · existing checkpoints only · no training',y=1.005,fontsize=14)
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(a.output/'answer_bias_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
