"""Plot completed effects with explicit restore/injection references; no new inference."""
import argparse,csv,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs=3,type=Path,required=True);p.add_argument('--analyses',nargs=3,type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
 from analyze_interventions import clustered_effects
 a.output.mkdir(parents=True,exist_ok=False);allpoints=[];fig,axes=plt.subplots(3,2,figsize=(13,12))
 for i,(label,run,analysis) in enumerate(zip(['FFT','LoRA','SSU-Wanda'],a.runs,a.analyses)):
  rows=list(csv.DictReader((analysis/'conditional_effects.csv').open()))
  restore=[r for r in rows if r['mode']=='restore' and float(r['alpha'])==0]
  def read(c):return [json.loads(x) for x in (run/c/'predictions.jsonl').open()]
  inject=[]
  for c in ['inject_L14_attention','inject_L14_mlp']:
   for r in clustered_effects(read(c),read('base'),a.bootstrap):r.update(condition=c,component=c.rsplit('_',1)[1]);inject.append(r)
  for j,data in enumerate([restore,inject]):
   conditions=list(dict.fromkeys(r['condition'] for r in data));ax=axes[i,j]
   for k,c in enumerate(conditions):
    def choose(task,lang,metric):return next(r for r in data if r['condition']==c and r['task']==task and r['input_language']==lang and r['output_language']==lang and r['metric']==metric)
    x=choose('knowledge_choice','sw','content_correct');y=choose('instruction_constraints','en','format_passed')
    xv=float(x['effect_final_minus_intervened_pp']);yv=float(y['effect_final_minus_intervened_pp'])
    def error(r,v):return [[max(0,v-float(r['ci95_low_pp']))],[max(0,float(r['ci95_high_pp'])-v)]]
    component=x['component'];mark='o' if component=='attention' else '^' if component=='mlp' else 's'
    ax.errorbar(xv,yv,xerr=error(x,xv),yerr=error(y,yv),fmt=mark,alpha=.55,capsize=3,label=c)
    allpoints.append(dict(model=label,reference='Final minus restore' if j==0 else 'Inject minus Base',condition=c,component=component,sw_effect_pp=xv,sw_ci_low_pp=x['ci95_low_pp'],sw_ci_high_pp=x['ci95_high_pp'],en_effect_pp=yv,en_ci_low_pp=y['ci95_low_pp'],en_ci_high_pp=y['ci95_high_pp'],sw_samples=x['samples'],sw_clusters=x['clusters'],en_samples=y['samples'],en_clusters=y['clusters']))
   ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6);ax.set_title(label+(': Final - restore(G)' if j==0 else ': inject(G) - Base'));ax.set_xlabel('SW knowledge choice (pp)');ax.set_ylabel('EN synthetic constraint pass (pp)');ax.grid(alpha=.15)
   if i==0:ax.legend(fontsize=6,loc='best',framealpha=.7)
 fig.suptitle('Predeclared development interventions: references shown separately\nPaired cluster bootstrap 95%; overlapping points are not jittered; no multiplicity correction',fontsize=12);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(a.output/'restore_vs_inject.png',dpi=160);fig.savefig(a.output/'restore_vs_inject.svg');plt.close(fig)
 with (a.output/'restore_vs_inject.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(allpoints[0]));w.writeheader();w.writerows(allpoints)
 (a.output/'protocol.json').write_text(json.dumps(dict(scope='Plot-only reanalysis of frozen development outputs. Reading/math and every metric remain in full conditional_effect tables. Not formal benchmark.',restoration='Final minus literal alpha=0 restore, including matched random removal controls',injection='Inject minus Base, not Final minus inject',point_count=len(allpoints),bootstrap=a.bootstrap,math_floor='Original restricted content extraction has floor effects; not displayed as if true math accuracy'),indent=2)+'\n');print(json.dumps({'points':len(allpoints),'output':str(a.output)}))
if __name__=='__main__':main()
