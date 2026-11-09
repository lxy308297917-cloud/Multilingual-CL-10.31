from pathlib import Path
import hashlib,json,yaml,datetime
R=Path('/path/to/project')
C=R/'configs/function_aware_formal_20261109';C.mkdir(parents=True,exist_ok=True)
O=Path('/path/to/storage/checkpoints/function_aware_formal_20261109');O.mkdir(parents=True,exist_ok=True)
L=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109');L.mkdir(parents=True,exist_ok=True)
Q=R/'research/function_aware_formal_20261109';Q.mkdir(parents=True,exist_ok=True)
jobs={};audit={}
for lang in ['sw','ar','ru']:
 if lang=='sw':
  source=R/'configs/functional_representation_20261109/p2/all_layer_formal_v1.yaml';lock=R/'configs/aya_4b_20261105/data.lock.json';old=Path('/path/to/storage/checkpoints/aya_4b_20261105/lora_final_v1')
 else:
  source=R/f'configs/aya_4b_generalization_20261107/{lang}_lora_final.yaml';lock=source.parent/f'data_{lang}.lock.json';old=Path('/path/to/storage/checkpoints/aya_4b_generalization_20261107')/f'{lang}_lora_final_v1'
 cfg=yaml.safe_load(source.read_text());exposure=old/'exposure-rank0.jsonl'
 assert exposure.exists() and (old/'adapter_model.safetensors').exists()
 audit[lang]={'source_config':str(source),'old_microbatch':cfg['per_device_train_batch_size'],'old_accumulation':cfg['gradient_accumulation_steps'],'original_adapter':str(old),'original_adapter_sha256':hashlib.sha256((old/'adapter_model.safetensors').read_bytes()).hexdigest(),'data_lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),'examples':sum(1 for _ in exposure.open()),'data_dir':cfg['dataset_dir']}
 assert audit[lang]['examples']==6400
 cfg.update(per_device_train_batch_size=2,gradient_accumulation_steps=8,save_strategy='steps',save_only_model=False,max_steps=400,save_steps=50)
 for method in ['all','full','kd','random','function']:
  if lang=='sw' and method=='all':continue
  name=f'{lang}_{method}';jobs[name]={'language':lang,'method':method,'data_lock':str(lock),'source_exposure':str(exposure),'steps':400,'status':'PREPARED_NOT_STARTED'}
  for stage,steps in [('smoke',2),('final',400)]:
   v=dict(cfg,max_steps=steps,output_dir=str(O/(name+'_'+stage+'_v1')),logging_dir=str(L/name/(stage+'-trainer')))
   target=C/(name+'_'+stage+'.yaml');
   if target.exists(): assert yaml.safe_load(target.read_text())==v
   else: target.write_text(yaml.safe_dump(v,sort_keys=False))
(C/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n');(Q/'reuse_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
stop=R/'research/crosslayer_formal_20261109/STOPPED_BY_USER.json'
stop.write_text(json.dumps({'status':'STOPPED_BY_USER_SUPERSEDED','at':datetime.datetime.now().astimezone().isoformat(),'reason':'Previous forwarded plan mistaken; do not resume crosslayer/SST/FFT family','replacement':'function_aware_formal_20261109','preserve_existing_artifacts':True,'partial_kl_latest_checkpoint':50,'partial_kl_is_not_Final400':True},indent=2)+'\n')
(Q/'current_objective.md').write_text('''# Current authorized objective\n\nComplete SW/AR/RU five-method fixed-Final400 formal comparisons from the corrected latest ChatGPT plan. This supersedes the previous conditional P3 goal and mistaken crosslayer solver family. No performance gate, no new seeds, no formal-test-derived weights.\n\n## Necessary protocol alignment\n\nExisting SW ordinary LoRA micro2/GA8 is reusable. AR/RU old ordinary LoRA micro1/GA16 are historical calibration adapters and will receive aligned micro2/GA8 control reruns.\n\nOld uniform representation uses 36 block endpoints, which cannot distinguish Attention/MLP weights. New common feature set has 72 sites: post-attention residual (input to post_attention_layernorm), and post-MLP residual (decoder block output), at the same protected answer positions on the exact same EN build128. Six anatomical groups each contain 12 sites. All uniform/random/function runs use these same features; uniform weights1 is the exact new objective baseline. The old SW36-site FullRep result remains separately labeled and is not passed off as the new common-feature control. This requires 14 new runs, rather than the provisional11.\n\nR_g = mean normalized squared feature drift over the12 sites in group g, divided by the fixed capability scale; total preservation = (1/6) sum_g w_g R_g. All six weights average1, bounded[0.5,2]. No layers frozen. Same target order, optimizer, old example order, teacher Base and scalar coefficient1.\n\nFunction weights: fixed attenuation0.75 and matched-total-effective-update-norm uniform controls. On build-only EN math/instruction and target reading/summary CE, risk = mean(old relative recovery compared with the matched uniform control) - max(0, mean(target relative cost compared with matched uniform)). Relative differences divide each reference NLL by max(NLL,0.1). Let z=max(risk,0); if z.sum=0 all weights1, otherwise w = 0.5 + 3*z/z.sum, then Euclidean project onto sum6 and [0.5,2]. Same rule across languages; calibration is part of algorithm, not an effectiveness screen. Random control uses NumPy default_rng(61415).permutation(6), same multiset. Freeze before new formal tests.\n\nThe reference calibration adapters are the original per-language completed LoRAs; their microbatch differs from new training but no new formal outcome is used to fit the weights. Both legacy and aligned ordinary baselines will be labeled explicitly.\n\nFormal target: Belebele, XL-Sum, official Global-MMLU and XNLI where available. EN: GSM8K, official IFEval strict, MMLU, HumanEval+, Belebele, XNLI. Fixed Final400, no test-based selection; report negative results and paired intervals. Additional resources and calibration compute counted separately.\n''')
print(json.dumps({'new_training_jobs':len(jobs),'audit':audit},indent=2))
