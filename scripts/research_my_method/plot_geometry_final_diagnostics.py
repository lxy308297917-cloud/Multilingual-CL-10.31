"""Static research figure from completed read-only audits, no new experiment."""
import argparse, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p = argparse.ArgumentParser()
for key in ['summary', 'precision', 'output']:
    p.add_argument('--'+key, type=Path, required=True)
a = p.parse_args(); assert not a.output.exists(); a.output.mkdir(parents=True)
data = json.loads(a.summary.read_text()); precision = json.loads(a.precision.read_text())
assert precision['metadata']['step'] in [20,400] and precision['all310_master_hashes_verified']
step=precision['metadata']['step']; final=step==400
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'axes.titleweight': 'bold'})
fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), constrained_layout=True)
names = [x['label'] for x in data]; colors = ['#2866a5', '#338970', '#788594']
y = [x['observed_over_null'] for x in data]
axes[0].bar(names, y, color=colors, width=.58)
axes[0].axhline(1, color='#555555', lw=1, ls='--')
for i, value in enumerate(y): axes[0].text(i, value+.065, f'{value:.3f}', ha='center')
axes[0].set(ylim=(0, 1.25), ylabel='Observed free energy / orientation reference',
            title='A  Existing SW Final updates')
axes[0].text(0, -.16, 'Fixed Base r99 space\nAnalytic isotropic reference; no p-value',
             transform=axes[0].transAxes, fontsize=9, va='top')
y = [100*x['outside_target_delta_energy_fraction'] for x in data]
axes[1].bar(names, y, color=colors, width=.58)
for i, value in enumerate(y): axes[1].text(i, value+2, f'{value:.1f}%', ha='center')
axes[1].set(ylim=(0, 100), ylabel='Share of total update squared norm (%)',
            title='B  Outside the 196 SST matrices')
axes[1].text(0, -.16, 'Includes tied embedding/head\nEnergy does not establish causality',
             transform=axes[1].transAxes, fontsize=9, va='top')
t = precision['totals']; keys = ['fp32_master', 'bf16_export']
free = [t[k]['target']['free_squared_norm'] for k in keys]
outside = [t[k]['target']['outside_free_squared_norm'] for k in keys]
axes[2].bar(['FP32 master', 'BF16 export'], free, color='#2866a5', width=.58, label='In free space')
axes[2].bar(['FP32 master', 'BF16 export'], outside, bottom=free, color='#dd9142', width=.58,
            label='Outside free space')
axes[2].set(ylabel='Squared Frobenius distance from Base',
            title=('C  NoRefresh Final400' if final else 'C  R10 engineering step20'))
height=max(x+y for x,y in zip(free,outside)); offset=height*.035
for i,k in enumerate(keys):
    fraction=100*t[k]['target']['outside_free_energy_fraction']
    label='Outside <0.00001%' if fraction<.00001 else f'Outside {fraction:.2f}%'
    axes[2].text(i,free[i]+outside[i]+offset,label,ha='center',fontsize=9)
axes[2].set_ylim(0,height*1.45)
axes[2].legend(loc='upper right', frameon=False, fontsize=8)
fig.suptitle('Parameter geometry audits — no causal or task-performance conclusion', fontsize=13)
for suffix in ['png', 'pdf']:
    fig.savefig(a.output/(('geometry_final_diagnostics.' if final else 'geometry_early_diagnostics.')+suffix), dpi=180,
                metadata={'Creator': 'Multilingual-CL research audit'})
plt.close(fig)
manifest = {'source_sha256': {key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in [('summary', a.summary), ('precision', a.precision)]},
            'files_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in a.output.iterdir()},
            'source_step': step, 'actual_final_geometry': final, 'functional_claim': False}
(a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
