from common import *
import datetime,yaml,subprocess
def main():
 assert not (R/'TRAINING_LOCK.json').exists()
 smoke=read(CK/'T2-smoke-v1/TRAINING_DONE.json');verified=read(CK/'T2-smoke-v1/RELOAD_VERIFIED.json')
 assert verified['weights_sha256']==smoke['weights_sha256']
 throughput=smoke['counts']['input_tokens']/smoke['training_seconds']
 projected=19135489/throughput*3/3600
 available=(datetime.datetime.fromisoformat(read(R/'run_manifest.json')['deadline_shanghai'])-datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))).total_seconds()/3600
 budget=dict(at=now(),selected='FULL-65K',samples=65415,steps=4089,effective_batch=16,seed=42,estimated_serial_training_hours=projected,training_contingency_fraction=.10,core_common_evaluation_reserved_hours=2.,analysis_gpu_reserved_hours=1.,packaging_reserved_hours=1.,total_reserved_hours=projected*1.1+4,available_hours=available,basis='Actual T2 most expensive8step smoke,39483 input tokens,plus prior500step36layer keep run. Single available3090; do not assume other users release GPUs.',historical_8k_baseline_reused=False,reason_not_reused='Different keep scales from original S3 audit, and full65k budget; all3 freshly initialized.',switch_rule='Budget frozen before any new formal scores. No result-dependent extension or truncation; unexpected interruption reported partial.')
 assert budget['total_reserved_hours']<available,budget
 write(R/'records/BUDGET_FROZEN.json',budget)
 keep=read(W/'work/s3-align-v0-20261113/keep_audit/KEEP_CONTRACT.json')
 files=[R/'code/common.py',R/'code/train_mapping.py',R/'assets/mappings.safetensors',R/'assets/mapping_metadata.json',R/'assets/gradient_calibration.json',R/'inputs/alignment.jsonl',W/'work/s3-align-v0-20261113/keep_loss.py',Path(keep['implementation']['module']),Path(keep['reference']['pool']),Path(keep['reference']['cache']),Path(keep['reference']['manifest']),DATA/'manifest.json',DATA/'ids.json']
 files += list((R/'configs').glob('T*-full-v1.yaml'))
 files += list((DATA/'sw_train_tokenized').rglob('*.arrow'))+list((DATA/'sw_train_tokenized').rglob('*.json'))
 base=read(PREV/'protocol.json')['base_identity']
 write(R/'TRAINING_LOCK.json',dict(at=now(),budget=budget,files={str(p):sha(p) for p in files},base_identity=base,calibration=read(R/'assets/gradient_calibration.json'),readout='L18 raw block fulluserbody mean',teacher='frozen Base EN single',keep=dict(mu=1,scales=keep['loss']['normalizers'],sites=36,positions='up to4 reference-answer prediction states',exposures_per_optimizer_step=1),primary_comparisons=['T2-T0','T2-T1','T1-T0'],screening_flag='SWreading+2pp and GSM8K/IFEval each>=T0-1pp; descriptive engineering flag only',scientific_warning='T1 differs from historical S3 in teacher/centering/scales; new arms isolate mapping, not historicalE-vsT2. One seed.'))
 evalfiles=[R/'code/evaluate.py',R/'code/common.py',P/'scripts/sft_v1/harness.py',P/'configs/sft_v1/eval_tasks/gsm8k.yaml',P/'configs/sft_v1/eval_tasks/ifeval.yaml',Path('/srv/storage/datasets/function-aware-formal-20261109/evaluation-v1/tasks/common/belebele_swh_Latn_locked.yaml'),Path('/srv/storage/datasets/function-aware-formal-20261109/evaluation-v1/tasks/common/_belebele_locked_base.yaml')]
 write(R/'EVALUATION_LOCK.json',dict(at=now(),files={str(p):sha(p) for p in evalfiles},required=['SWBelebele900ACC','ENGSM8K1319flexible-extract','ENIFEval541prompt_strict'],batch='Reading upstreamHF auto:4; core officialvLLM32,eager,.82 memory,8192 context',source='Nov15executionv4 unchanged upstream prompt,scoring,generation parameters; new result directories and model identities',optional='Only common full groups; knowledge/translation/code secondary',base_policy='Reuse only after full execution/scientific identity comparison or rerun Base',scores_not_read_before_freeze=True))
 manifest=read(R/'run_manifest.json');manifest.update(status='CONFIG_FROZEN_READY_TO_TRAIN',budget='FULL-65K',training_lock_sha256=sha(R/'TRAINING_LOCK.json'),evaluation_lock_sha256=sha(R/'EVALUATION_LOCK.json'));manifest['tasks']['1']='CACHED_METRICS_COMPLETE_PLOT_PENDING';manifest['tasks']['3']='HISTORICAL_TRANSITIONS_COMPLETE_NEW256_FROZEN';write(R/'run_manifest.json',manifest)
 print(json.dumps(budget,ensure_ascii=False))
if __name__=='__main__':main()
