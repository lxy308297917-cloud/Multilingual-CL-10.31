"""Aggregate actual exposure, identity and computation; no per-item export."""
import argparse,hashlib,json,struct,collections
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--reference100',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run=a.project/'results/decision_stability';assert not a.output.exists();a.output.mkdir(parents=True)
    ref=json.loads((a.reference100/'accounting-rank0.json').read_text())['global_counts']
    training={};aux_reference=None;all_main_exact=True;main_gpu_hours=0
    for label,name in [('B','B-marginal100-v1'),('C','C-content100-v1'),('D','D-replay100-v1')]:
        result=run/name;done=json.loads((result/'COMPLETED.json').read_text());launch=json.loads((result/'launch.json').read_text());model=Path(done['output'])
        assert done['steps']==100 and done['exposure_exact'] and done['original_schedule_steps']==400
        state=json.loads((model/'trainer_state.json').read_text());assert state['global_step']==100 and state['max_steps']==400
        ranks=[];exposure={};aux_rows=[]
        for rank in [0,1]:
            account=json.loads((model/f'accounting-rank{rank}.json').read_text());assert account['global_counts']==ref
            current=model/f'exposure-rank{rank}.jsonl';old=a.reference100/f'exposure-rank{rank}.jsonl';assert [json.loads(v) for v in current.open()]==[json.loads(v) for v in old.open()]
            exposure[str(rank)]=dict(rows=800,sha256=sha(current),all_fields_identical_to_A100=True,serialization_bytewise_identical=current.read_bytes()==old.read_bytes())
            rows=[json.loads(s) for s in (model/f'aux-rank{rank}.jsonl').open()];assert len(rows)==100
            assert [v['update_step'] for v in rows]==list(range(1,101))
            signature=[(v['semantic_id'],v['language'],v['input_tokens'],v['rotations']) for v in rows]
            if aux_reference is None:aux_reference=signature
            assert signature==aux_reference
            assert all(v['hf_accumulation_factor']==1 and v['model_accepts_loss_kwargs'] and v['num_items_present'] for v in rows)
            if label=='D':assert all(v['replay_denominator']=='full vocabulary' for v in rows)
            aux_rows.append(rows);ranks.append(dict(peak_allocated=account['max_memory_allocated'],peak_reserved=account['max_memory_reserved']))
        aux=aux_rows[0];gpu_hours=2*done['seconds']/3600;main_gpu_hours+=gpu_hours
        final_identity=json.loads((run/('formal-'+label+'100-v1')/'launch.json').read_text())['model_sha256']
        dtype_counts=collections.Counter()
        for filename in final_identity:
            with (model/filename).open('rb') as f:
                size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
            dtype_counts.update(v['dtype'] for k,v in header.items() if k!='__metadata__')
        training[label]=dict(final_model_weight_sha256=final_identity,final_export_tensor_dtype_counts=dict(dtype_counts),optimizer_steps=100,original_schedule_steps=400,world_size=2,main_exposure=ref,
            main_trace_identity=exposure,aux_unique_semantic_ids=len(set(v['semantic_id'] for v in aux)),aux_language_questions=100,
            aux_logical_forward_sequences=sum(v['rotations'] for v in aux),aux_physical_forward_sequences=sum(v['rotations'] for rows in aux_rows for v in rows),
            aux_logical_input_tokens=sum(v['input_tokens'] for v in aux),aux_physical_input_tokens=sum(v['input_tokens'] for rows in aux_rows for v in rows),
            aux_questions_by_language={l:sum(v['language']==l for v in aux) for l in ['en','sw']},
            training_process_wall_seconds=done['seconds'],actual_main_samples_per_wall_second=ref['samples']/done['seconds'],actual_main_input_tokens_per_wall_second=ref['input_tokens']/done['seconds'],gpu_wall_hours_proxy=gpu_hours,prerequisite_probe_seconds_excluded=done.get('prerequisite_engineering_seconds_excluded',0),
            peak_memory_by_rank=ranks,source_weight_sha256=launch['source_weights'],config_sha256=launch['config_sha256'],source_code_sha256=launch['code_sha256'],
            teacher_sha256=launch['teacher_sha256'],data_protocol_sha256=launch['data_protocol_sha256'],
            auxiliary_objective_first10_mean=float(np.mean([v['auxiliary_loss'] for v in aux[:10]])),auxiliary_objective_last10_mean=float(np.mean([v['auxiliary_loss'] for v in aux[-10:]])),
            warning='Auxiliary objectives have different meanings/scales; do not rank methods by training objective. Loss is added once per update. Same question duplicated across ranks is averaged by distributed gradients.')
    teacher=json.loads((run/'teacher-w1/DONE.json').read_text());engineering={}
    for name in ['engineering-content-2-v1','engineering-replay-full-vocab-2-v1']:
        done=json.loads((run/name/'COMPLETED.json').read_text());assert done['steps']==2 and done['exposure_exact'];engineering[name]=dict(optimizer_steps=2,main_samples=32,seconds=done['seconds'],gpu_wall_hours_proxy=2*done['seconds']/3600)
    formal={k:json.loads((run/('formal-'+k+'100-v1')/'DONE.json').read_text()) for k in ['A','B','C','D']}
    fresh={}
    for name in ['p0-w1-dev','p0-a100-dev']+[f'final-{m}-holdout' for m in ['W1','A','B','C','D']]+[f'final-{m}-dev' for m in ['B','C','D']]:
        fresh[name]=json.loads((run/name/'DONE.json').read_text())
    total=dict(new_scientific_training_gpu_wall_hours_proxy=main_gpu_hours,engineering_gpu_wall_hours_proxy=sum(v['gpu_wall_hours_proxy'] for v in engineering.values()),
        teacher_physical_gpu_wall_hours=teacher['gpu_wall_hours'],fresh_scoring_gpu_wall_hours=sum(v['gpu_wall_hours'] for v in fresh.values()),
        formal_reporting_gpu_wall_hours_proxy=sum(v['seconds'] for v in formal.values())/3600)
    total['all_new_gpu_wall_hours_proxy']=sum(total.values())
    payload=dict(status='COMPLETE',A=dict(final_model_weight_sha256=json.loads((run/'formal-A100-v1/launch.json').read_text())['model_sha256'],reused_existing_checkpoint100=True,new_training_seconds=0,main_exposure=ref,reference_reuse_verified=True),
        training=training,engineering=engineering,teacher=dict(physical_cost=teacher,selected_semantic_ids=53,actual_used_semantic_ids=50,
            same_teacher_budget_per_aux_arm_gpu_hours=teacher['gpu_wall_hours'],allocated_shared_physical_gpu_hours_per_arm=teacher['gpu_wall_hours']/3,
            note='Single offline cache physically computed once; full nominal teacher budget shown equally for B/C/D. A has no auxiliary or teacher compute; four arms are not total-compute matched.'),
        formal=formal,fresh_scoring=fresh,total=total,validation=dict(all_HA_main_exposure_exact=True,all_BCD_auxiliary_schedule_exact=True,full_vocabulary_replay=True,
            actual_HF_loss_kwargs_normalization_verified=True,formal100_and_original400_scheduler=True,raw_questions_or_predictions_exported=False),
        limits=['Process wall time times allocated GPU count includes loading/checking/saving; not kernel-time or FLOPs.','Input token counters count data exposure, not activation-checkpoint recomputation, padded kernel work or all engineering oracle calls.','GPU pair/topology and cache effects differ; do not claim algorithm speed ranking from wall times.','Training-side auxiliary losses are not downstream effectiveness evidence.','Native Trainer scheduled6400/400 but stopped100; its configured-epoch throughput can overstate actual throughput. Reported exposure rates here use actual1600 samples.'])
    (a.output/'cost_and_validation.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(total))
if __name__=='__main__':main()
