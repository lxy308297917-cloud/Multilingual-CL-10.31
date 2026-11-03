"""Summarize actual completed SST logs without interpreting cost as efficacy."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();a.output.mkdir(parents=True)
load=lambda f:[json.loads(x) for x in f.read_text().splitlines() if x]
steps=load(a.model/'step-cost.jsonl');mat=load(a.model/'matrix-updates.jsonl');svd=load(a.model/'svd-cost.jsonl')
receipt=json.loads(a.receipt.read_text()); inventory=json.loads((a.model/'subspace-inventory.json').read_text())
assert receipt['exit_code']==0 and receipt['global_step']==400 and len(steps)==400
assert [x['step'] for x in steps]==list(range(1,401))
names=set(inventory['matrices']);assert len(names)==196 and len(mat)==400*196
by={}
for row in mat:
    by.setdefault(row['step'],{})[row['name']]=row
assert set(by)==set(range(1,401)) and all(set(x)==names for x in by.values())
free=sum(x['free_space_dimension'] for x in inventory['matrices'].values())
target=sum(x['shape'][0]*x['shape'][1] for x in inventory['matrices'].values())
per=[]
for k,v in sorted(by.items()):
    raw=sum(x['raw_adam_update_norm']**2 for x in v.values())**.5
    applied=sum(x['applied_update_norm']**2 for x in v.values())**.5
    per.append({'step':k,'raw_optimizer_proposal_norm':raw,'applied_target_update_norm':applied,
        'applied_over_raw_norm':(applied/raw if raw>0 else None),'numerically_changed_target_elements':sum(x['numerically_changed_master_elements'] for x in v.values()),
        'max_current_left_relative':max(x['current_reference_leakage']['left_relative'] for x in v.values()),
        'max_current_right_relative':max(x['current_reference_leakage']['right_relative'] for x in v.values())})
def summary(vals):return {'min':min(vals),'median':statistics.median(vals),'mean':statistics.mean(vals),'max':max(vals)}
record={'status':'VERIFIED_COMPLETE','global_step':400,'matrix_updates_verified':len(mat),
 'trainable_unique_parameters':inventory['trainable_unique_parameters'],'target_coordinates':target,
 'outside_target_parameters':sum(x['elements'] for x in inventory['unconstrained_parameters']),
 'free_space_dimension':free,'free_space_coordinate_fraction':free/target,
 'actual_wall_seconds':receipt['wall_seconds'],'wall_seconds_per_optimizer_step':receipt['wall_seconds']/400,
 'native_train_results':json.loads((a.model/'train_results.json').read_text()),
 'resource_summaries':load(a.model/'training-resource-summary.jsonl'),
 'svd_events':svd,'optimizer_and_projection_seconds':summary([x['optimizer_and_projection_seconds'] for x in steps]),
 'total_seconds_including_refresh':summary([x['total_seconds_including_refresh'] for x in steps]),
 'target_applied_over_raw_norm':summary([x['applied_over_raw_norm'] for x in per if x['applied_over_raw_norm'] is not None]),
 'zero_raw_proposal_steps':[x['step'] for x in per if x['raw_optimizer_proposal_norm']==0],
 'target_changed_elements_per_step':summary([x['numerically_changed_target_elements'] for x in per]),
 'interpretation':['Norms cover only196 constrained matrices, not whole-model Adam proposal.',
 'Clipping/scheduler/Adam executed upstream before cumulative SST projection.',
 'First zero-LR step has raw norm0; at tiny final LR cumulative projection/roundoff can exceed the very small raw proposal.',
 'Logged callback timing excludes forward/backward and some save/startup time.',
 'Reused initial SVD cache is disclosed separately; no fabricated initial cost in this run.'],
 'sources_sha256':{name:hashlib.sha256((a.model/name).read_bytes()).hexdigest() for name in ['step-cost.jsonl','matrix-updates.jsonl','svd-cost.jsonl','subspace-inventory.json','train_results.json','training-resource-summary.jsonl']}}
(a.output/'summary.json').write_text(json.dumps(record,indent=2)+'\n');(a.output/'per_step.json').write_text(json.dumps(per,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','matrix_updates_verified','free_space_dimension','optimizer_and_projection_seconds','target_applied_over_raw_norm']}))
