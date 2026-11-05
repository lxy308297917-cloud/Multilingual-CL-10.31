"""33-record real-model canary tests partial batches, epoch saving and exposures."""
from profile_paths import resolve
import os, json, collections
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from common import *
from datasets import load_from_disk, DatasetDict
import yaml
src = DATA / 'training-v1/sw'
out = DATA / 'engineering33-v1'
assert not out.exists()
out.mkdir()
data = load_from_disk(str(src / 'tokenized'))['train']
indices = sorted(range(len(data)), key=lambda i: oh([42, 'engineering33', i]))[:33]
sub = data.select(indices)
DatasetDict({'train': sub}).save_to_disk(str(out / 'tokenized'))
raw = rows(src / 'sw_sft.jsonl')
(out / 'sw_sft.jsonl').write_text(''.join((json.dumps(raw[i], ensure_ascii=False) + '\n' for i in indices)))
(out / 'dataset_info.json').write_bytes((src / 'dataset_info.json').read_bytes())
fp = collections.defaultdict(list)
full = json.loads((src / 'training_data.json').read_text())
ids = []
for i, r in zip(indices, sub):
    id = full['source_ids'][i]
    ids.append(id)
    h = __import__('hashlib').sha256(json.dumps([r['input_ids'], r['labels']], separators=(',', ':')).encode()).hexdigest()
    fp[h].append(id)
dump(out / 'fingerprint_source_ids.json', fp)
dump(out / 'training_data.json', {'language': 'sw', 'samples': 33, 'source_ids': ids, 'source_manifest_sha256': sha(src / 'training_data.json'), 'purpose': 'ENGINEERING_ONLY: two true epochs, N33 remainder, not formal efficacy result'})
checkpoint = Path(resolve('CHECKPOINTS'))
checkpoint.mkdir(exist_ok=True)
link = Path(resolve('CHECKPOINTS'))
if not link.exists():
    link.symlink_to(checkpoint, target_is_directory=True)
assert link.resolve() == checkpoint
cfg = yaml.safe_load((ROOT / 'configs/unified_language_cl/mathoctopus/mathoctopus_single_es_seed42_v1.yaml').read_text())
cfg.update(dataset_dir=str(out), dataset='sw_sft', tokenized_path=str(out / 'tokenized'), num_train_epochs=2.0, max_steps=-1, learning_rate=5e-06, save_strategy='epoch', save_only_model=True, dataloader_drop_last=False, output_dir=str(checkpoint / 'engineering33_sw_seed42_v1'), logging_dir=resolve('LOGS') + '/engineering33/tensorboard', logging_steps=1)
cfg.pop('save_steps', None)
p = ROOT / 'configs/math_sft_base_fft/engineering33_sw_seed42_v1.yaml'
assert not p.exists()
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(p)
