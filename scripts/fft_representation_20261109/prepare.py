from pathlib import Path
import datetime,hashlib,json,yaml
R=Path('/path/to/project');C=R/'configs/fft_representation_20261109';Q=R/'research/fft_representation_20261109';K=Path('/path/to/storage/checkpoints/fft_representation_20261109');L=Path('/path/to/logs/qwen3-multilingual/fft-representation-20261109')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def main():
 for p in [C,Q,K,L]:p.mkdir(parents=True,exist_ok=True)
 old=Path('/path/to/storage/checkpoints/aya_4b_20261105/fft_final_v2');done=json.loads((old/'TRAINING_DONE.json').read_text());assert done['steps']==400 and done['samples']==6400
 for name,digest in done['weights_sha256'].items():assert sha(old/name)==digest,name
 source=R/'configs/aya_4b_20261105/fft_v2.yaml';cfg=yaml.safe_load(source.read_text());assert sha(source)==done['config_sha256']
 assert cfg['finetuning_type']=='full' and cfg['per_device_train_batch_size']==1 and cfg['gradient_accumulation_steps']==16 and cfg['learning_rate']==1e-5 and cfg['seed']==42
 lock=R/'configs/aya_4b_20261105/data.lock.json';assert sha(lock)==done['data_manifest_sha256']
 oldmanifest=json.loads(lock.read_text());assert all(sha(Path(cfg['dataset_dir'])/k)==v for k,v in oldmanifest['files'].items())
 ds=json.loads(Path(cfg['deepspeed']).read_text());assert ds['zero_optimization']['stage']==2 and ds['zero_optimization']['offload_optimizer']['device']=='cpu'
 cfg.update(use_reentrant_gc=False,gradient_checkpointing_kwargs={'use_reentrant':False},save_strategy='steps',save_only_model=False,max_steps=400,save_steps=100)
 longest=Path('/path/to/storage/datasets/functional-representation-20261109/p2-v1/engineering_longest16_tokenized');assert longest.is_dir()
 for mode in ['none','full36']:
  for stage,steps in [('smoke',3),('final',400)]:
   c=dict(cfg,max_steps=steps,output_dir=str(K/(mode+'_'+stage+'_v1')),logging_dir=str(L/mode/(stage+'-trainer')))
   if stage=='smoke':c['tokenized_path']=str(longest)
   path=C/(mode+'_'+stage+'.yaml');assert not path.exists();path.write_text(yaml.safe_dump(c,sort_keys=False))
 record=dict(status='HISTORICAL_FFT_VERIFIED_NEW_PAIRED_RUNS_PREPARED',at=datetime.datetime.now().astimezone().isoformat(),historical_fft=str(old),historical_done_sha256=sha(old/'TRAINING_DONE.json'),historical_weights_sha256=done['weights_sha256'],historical_seconds=json.loads((old/'accounting-rank0.json').read_text())['elapsed_this_process_seconds'],original_data_lock_sha256=sha(lock),new_modes=['none','full36'],new_target_recipe='same SW6400, original order, micro1 GA16, LR1e-5, AdamW CPU offload, cosine warmup0.05, seed42, BF16, cutoff2048, no packing, fixedFinal400',implementation_amendment='non-reentrant checkpointing plus a SINGLE combined backward per target microbatch; same change in protected and unprotected new FFT; historic FFT retained separately',protected_old_budget='Original P2 build128, original raw36 feature cache/scales/lambda1; eight old rows per optimizer step (every second target microbatch), exactly matching P2 old-sample exposure3200 rather than doubling it because FFT GA16',engineering='3 updates on frozen16 longest target examples and longest old examples; no benchmark scores; starts formal400 fresh only if masks, gradients, applied LR, single backward and memory pass',longest_target_input_sha256={str(p.relative_to(longest)):sha(p) for p in longest.rglob('*') if p.is_file()},old_scale_source='Historical unprotected SW LoRA, frozen P2 scales; no FFT-specific refitting')
 (Q/'protocol.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='longest_target_input_sha256'},indent=2))
 entry=Path('/path/to/models/checkpoint/fft_representation_20261109')
 if entry.exists() or entry.is_symlink():assert entry.resolve()==K.resolve()
 else:entry.symlink_to(K,target_is_directory=True)
 (Q/'README.md').write_text('''# FFT＋36层全表示保持补充实验\n\n依据用户2026-11-09上传的新FFT任务，优先补齐SW上的跨参数化验证。既有三语言LoRA任务保留，不覆盖。\n\n- 历史FFT完整400步、6400条和两个权重文件已重新核验，另列历史行。\n- 新增匹配普通FFT与FFT＋全表示保持，各完整400步。两组均采用非重入梯度检查点及单次联合反向传播，避免ZeRO2双反向的更新边界与重复梯度归约问题。改变实现的普通FFT也补跑，避免与保护效果混淆。\n- 保护组使用P2原始36层特征，不是新六组72点特征；同Base缓存、EN build128、固定尺度与lambda1，check128不训练。每步8条旧样本，曝光量与P2对齐。\n- 3步工程验证采用已有最长16条目标样本，检查显存、全参数梯度、非零表示损失、CPUAdam实用LR和单次反向。只检查代码，不读取benchmark；正式400步重新从Base开始。\n- 两组FFT完整测SW/EN阅读、SW摘要、SW知识、EN MMLU、GSM8K strict/flexible、IFEval prompt/instruction strict、XNLI SW/EN、HumanEval+及双向FLORES；历史完整分数仅在协议匹配时复用。翻译负结果保留。\n- 最终同时展示Base、历史FFT、匹配FFT、新FFT＋表示保持、P2 LoRA/FullRep/CE Replay、历史SSU，标明学习率、microbatch、可训练参数及额外成本差异。\n\n当前状态为PREPARED，不是完成结果。所有输出独立保存，不因正式分数早停、改系数或选checkpoint。\n''')
if __name__=='__main__':main()
