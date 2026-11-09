from pathlib import Path
import json, shutil, hashlib
import yaml
W=Path(__file__).parent
R=Path('/path/to/project')
S=R/'scripts/functional_representation_20261109/p2'
text=(S/'train.py').read_text()
text=text.replace("D = Path('/path/to/storage/datasets/functional-representation-20261109/p2-v1')", "D = Path('/path/to/storage/datasets/function-aware-formal-20261109/features-v1') if os.environ.get('FA_METHOD') in ['full','random','function'] else Path('/path/to/storage/datasets/functional-representation-20261109/p2-v1')")
text=text.replace("from losses import trainer_class", "from losses import trainer_class")
text=text.replace("R / 'configs/aya_4b_20261105/data.lock.json'", "Path(os.environ['FA_DATA_LOCK'])")
text=text.replace("original = Path('/path/to/storage/checkpoints/aya_4b_20261105/lora_final_v1/exposure-rank0.jsonl')", "original = Path(os.environ['FA_SOURCE_EXPOSURE'])")
text=text.replace("feature_layers=36 if trainer.old_mode == 'full_feature' else 0", "feature_layers=(0 if os.environ['FA_METHOD']=='kd' else 72) if trainer.old_mode == 'full_feature' else 0")
text=text.replace("mode=args.mode,", "mode=os.environ['FA_METHOD'], language=os.environ['FA_LANGUAGE'],")
# KD references are loaded directly in the objective; preserve the P2 entry contract with an empty cache.
text=text.replace("load_file(str(D / 'build_reference_features.safetensors')) if args.mode == 'full_feature' else None", "({x['id']: None for x in rows} if os.environ['FA_METHOD']=='kd' else load_file(str(D / 'build_reference_features.safetensors'))) if args.mode == 'full_feature' else None")
(W/'train.py').write_text(text)
shutil.copy2(R/'scripts/functional_risk_p3_20261109/logit_kd.py',W/'logit_kd.py')
loss=(S/'losses.py').read_text()
# Reuse the audited sequential backward and accumulation implementation, replacing only objective.
pos=loss.index('\ndef trainer_class')
(W/'trainer_core.py').write_text(loss[pos+1:].replace('def trainer_class','def trainer_class',1))
print('entry and audited backward prepared')
