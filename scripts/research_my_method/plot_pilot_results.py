"""Completed preregistered exploratory contrast and fixed-stage development curves."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
d=json.loads(a.analysis.read_text());assert d['status']=='VERIFIED_EXPLORATORY_COMPLETE' and d['predictions_verified']==5130
fig,axs=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
plt.rcParams.update({'pdf.fonttype':42})
rows=[x for x in d['contrasts'] if x['primary_exploratory']];assert len(rows)==2
for i,x in enumerate(rows):
 axs[0].errorbar(x['delta'],i,xerr=[[x['delta']-x['ci95_low']],[x['ci95_high']-x['delta']]],fmt='o',capsize=5,color=['#27836a','#2866a5'][i],markersize=7)
 axs[0].text(x['ci95_high']+.2,i,f"{x['delta']:+.2f} pp",va='center')
axs[0].axvline(0,color='#777',linestyle='--',linewidth=1)
axs[0].set(yticks=range(2),yticklabels=[x['language'].upper() for x in rows],ylim=(-.65,1.65),xlim=(-4,8),xlabel='Paired minus shuffled ACC (percentage points)',title='A  Prespecified 10% intervention')
axs[0].text(0,-.24,'n=114 semantic IDs; paired 95% bootstrap intervals\nBoth include zero; no established SW/HA benefit',transform=axs[0].transAxes,fontsize=9,va='top')
labels=['Base','SW-FFT-Final','SW-HA-FFT-Final']
for lang,color in zip(['en','sw','ha'],['#788594','#27836a','#2866a5']):
 vals=[100*d['sequential_dev'][name][lang]['acc'] for name in labels]
 axs[1].plot(range(3),vals,'o-',label=lang.upper(),color=color,lw=2)
 for i,y in enumerate(vals):axs[1].annotate(f'{y:.1f}',(i,y),xytext=(0,(-15 if (lang=='ha' and i==0) or (lang=='sw' and i==2) else 7)),textcoords='offset points',ha='center',fontsize=8)
axs[1].set(xticks=range(3),xticklabels=['Base','After SW','After Hausa'],ylabel='Development ACC (%)',ylim=(15,71),title='B  Fixed FFT stages, same development items')
axs[1].legend(frameon=False,loc='upper right')
axs[1].text(0,-.24,'0-shot, FP32/eager, exported weights; n=114/language\nExploratory and historically exposed dev, not formal test',transform=axs[1].transAxes,fontsize=9,va='top')
for ax in axs:
 ax.spines['right'].set_visible(False);ax.spines['top'].set_visible(False)
fig.suptitle('Finite mechanism pilot: no validated new training method',fontsize=13)
for ext in ['png','pdf']:fig.savefig(a.output/('pilot_results.'+ext),dpi=180)
plt.close(fig)
manifest={'source_sha256':hashlib.sha256(a.analysis.read_bytes()).hexdigest(),'predictions_verified':5130,'formal_benchmark':False,'files_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in a.output.iterdir()}}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
