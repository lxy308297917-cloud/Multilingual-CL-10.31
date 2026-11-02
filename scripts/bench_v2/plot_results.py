"""Publication figure of paired per-task changes, never a combined score."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 data=json.loads(a.results.read_text());specs=[('flores200_en_sw_full_devtest_v2','chrF++','MT: EN → SW','chrF++ points'),('flores200_sw_en_full_devtest_v2','chrF++','MT: SW → EN','chrF++ points'),('xlsum_sw_v2','ROUGE-Lsum','Summarization: SW → SW','ROUGE-Lsum points'),('xlsum_en_v2','ROUGE-Lsum','Summarization: EN → EN','ROUGE-Lsum points')];comparisons={(x['benchmark'],x['method'],x['metric']):x for x in data['paired_comparisons']};fig,axes=plt.subplots(2,2,figsize=(11,6.7));colors=['#b45309','#2563eb','#0f766e']
 for ax,(task,metric,title,unit) in zip(axes.flat,specs):
  ax.axvline(0,color='#9ca3af',lw=1,ls='--');n=None
  for y,(method,label,color) in enumerate(zip(['full','lora','ssu'],['FFT','LoRA','SSU-Wanda'],colors)):
   c=comparisons.get((task,method,metric))
   if c is None:continue
   n=c['samples'];ax.hlines(y,c['ci95_low'],c['ci95_high'],color=color,lw=2.5);ax.plot(c['delta'],y,'o',color=color,ms=7)
  ax.set_yticks([0,1,2],['FFT','LoRA','SSU-Wanda']);ax.set_ylim(2.55,-.55);ax.set_title(title+(f'  (n={n:,})' if n else '  (pending)'),loc='left',fontsize=11);ax.set_xlabel('Change from Base ('+unit+')',fontsize=9);ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Fixed Final checkpoints: paired changes in generation benchmarks',fontsize=14,y=.99);fig.text(.5,.015,'Points: fine-tuned minus Base; lines: 95% paired bootstrap intervals (1,000 resamples, seed 42; unadjusted).\nEach panel is a separate metric and dataset. Code execution scores are NOT_RUN.',ha='center',fontsize=8.5,color='#374151');fig.tight_layout(rect=(0,.075,1,.95));a.output.mkdir(parents=True);fig.savefig(a.output/'paired_changes.svg',bbox_inches='tight');fig.savefig(a.output/'paired_changes.png',dpi=180,bbox_inches='tight');print(a.output)
if __name__=='__main__':main()
