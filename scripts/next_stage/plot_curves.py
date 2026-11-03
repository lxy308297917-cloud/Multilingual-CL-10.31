from pathlib import Path
import argparse,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);d=json.loads(a.analysis.read_text());lookup={(x['group'],x['step'],x['task'],x['language']):x for x in d['rows']};colors={'ha_only':'#2678a5','fft':'#ba574c','sst':'#4d9276'};labels={'ha_only':'Base -> HA FFT','fft':'SW FFT -> HA FFT','sst':'SW SST -> HA SST'};steps=[0,100,200,300,400]
fig,axes=plt.subplots(2,3,figsize=(14,7.5));specs=[('reading_exposed','ha','acc','HA reading (exposed180) ACC %'),('knowledge','ha','acc','HA knowledge (exposed114) ACC %'),('heldout_assistant_nll','ha','token_weighted_nll','HA heldout assistant-token NLL'),('reading_exposed','sw','acc','SW reading (exposed180) ACC %'),('knowledge','sw','acc','SW knowledge (exposed114) ACC %')]
for ax,(task,lang,metric,title) in zip(axes.flat,specs):
 for group in d['groups']:
  y=[lookup[group,s,task,lang][metric]*(100 if metric=='acc' else 1) for s in steps];ax.plot(steps,y,'-o',color=colors[group],label=labels[group],ms=4)
 ax.set_title(title,fontsize=11);ax.set_xlabel('HA optimizer step');ax.set_xticks(steps);ax.grid(alpha=.2)
ax=axes.flat[-1]
for group in d['groups']:
 xs=[x['step'] for x in d['training_loss'] if x['group']==group];ys=[x['loss'] for x in d['training_loss'] if x['group']==group];ax.plot(xs,ys,color=colors[group],label=labels[group],lw=1)
ax.set_title('Native logged training loss (auxiliary)',fontsize=11);ax.set_xlabel('HA optimizer step');ax.grid(alpha=.2)
fig.suptitle('Frozen HA learning curves: all stage points, fixed Final400',fontsize=15);handles,labs=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labs,ncol=3,loc='lower center',bbox_to_anchor=(.5,.038),frameon=False)
fig.text(.5,.012,'One seed; exposed development, not pristine confirmation. Different SW initial states. NLL is auxiliary; no checkpoint selection.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.10,1,.94])
for ext in ['png','pdf']:fig.savefig(a.output/('ha_learning_curves.'+ext),dpi=180)
(a.output/'manifest.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(a.analysis.read_bytes()).hexdigest(),'groups':d['groups'],'metric_or_data_changed':False,'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
