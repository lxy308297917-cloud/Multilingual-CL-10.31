"""Verify true epoch coverage and padding without dropping nondivisible examples."""
import argparse, collections, json, math
from pathlib import Path
from common import *
from safetensors import safe_open
import torch
torch.set_num_threads(4)
import yaml
ap = argparse.ArgumentParser()
ap.add_argument('--config', type=Path, required=True)
a = ap.parse_args()
cfg = yaml.safe_load(a.config.read_text())
out = Path(cfg['output_dir'])
data = Path(cfg['dataset_dir'])
info = json.loads((data / 'training_data.json').read_text())
fps = json.loads((data / 'fingerprint_source_ids.json').read_text())
expected = collections.Counter({k: len(v) for k, v in fps.items()})
n = info['samples']
assert sum(expected.values()) == n
base = {}
for p in sorted(MODEL.glob('*.safetensors')):
    with safe_open(str(p), framework='pt', device='cpu') as f:
        for k in f.keys():
            base[k] = f.get_slice(k).get_shape()
prior = collections.Counter()
reports = []
for ep in [1, 2]:
    step = ep * math.ceil(n / 16)
    p = out / f'checkpoint-{step}'
    marker = json.loads((p / 'EPOCH_CHECKPOINT.json').read_text())
    assert marker['epoch'] == ep and marker['step'] == step
    trace = []
    for rank in [0, 1]:
        account = json.loads((p / f'accounting-rank{rank}.json').read_text())
        rs = rows(p / f'exposure-rank{rank}.jsonl')
        assert account['trace_covers_all_local_samples'] and len(rs) == account['local_counts']['samples']
        assert account['step'] == step
        trace += rs
    got = collections.Counter((x['input_label_sha256'] for x in trace))
    assert not prior - got, ('cumulative trace was reset', ep)
    current = got - prior
    assert not expected - current, ('missing source coverage', ep, expected - current)
    assert not set(current) - set(expected)
    extras = sum(current.values()) - n
    assert 0 <= extras < 4, (ep, extras)
    shapes = {}
    weights = {}
    for f in sorted(p.glob('*.safetensors')):
        weights[f.name] = sha(f)
        with safe_open(str(f), framework='pt', device='cpu') as r:
            for k in r.keys():
                assert k not in shapes
                shapes[k] = r.get_slice(k).get_shape()
    missing = set(base) - set(shapes)
    assert missing in [set(), {'lm_head.weight'}] and (not set(shapes) - set(base)), missing
    if missing:
        assert json.loads((p / 'config.json').read_text())['tie_word_embeddings'] and base['lm_head.weight'] == shapes['model.embed_tokens.weight']
        shapes['lm_head.weight'] = shapes['model.embed_tokens.weight']
    assert shapes == base and weights == marker['weights_sha256']
    sampled_update = {}
    for key in ['model.layers.0.self_attn.q_proj.weight', 'model.layers.14.mlp.down_proj.weight', 'model.norm.weight']:
        ts = []
        for folder in [MODEL, p]:
            for f in folder.glob('*.safetensors'):
                with safe_open(str(f), framework='pt', device='cpu') as file:
                    if key in file.keys():
                        ts.append(file.get_tensor(key).float())
                        break
            else:
                raise RuntimeError('Missing sampled tensor ' + key)
        delta = ts[1] - ts[0]
        sampled_update[key] = {'elements': delta.numel(), 'changed_elements': int((delta != 0).sum()), 'l2_difference': float(torch.linalg.vector_norm(delta))}
    assert any((x['changed_elements'] for x in sampled_update.values())), 'No actual update in sampled parameter tensors'
    from transformers import AutoModelForCausalLM
    loaded = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map='cpu', local_files_only=True, low_cpu_mem_usage=True)
    loaded.eval()
    assert loaded.lm_head.weight.data_ptr() == loaded.model.embed_tokens.weight.data_ptr()
    with torch.no_grad():
        logits = loaded(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits
    assert torch.isfinite(logits).all() and logits.shape[1] == 3
    del loaded, logits
    report = {'status': 'VALIDATED', 'epoch': ep, 'steps': step, 'source_records_per_epoch': n, 'epoch_exposure_samples': sum(current.values()), 'epoch_extra_distributed_padding': extras, 'every_source_example_covered': True, 'cumulative_samples': len(trace), 'cumulative_input_tokens': sum((x['input_tokens'] for x in trace)), 'cumulative_shifted_supervised_tokens': sum((x['shifted_assistant_tokens'] for x in trace)), 'trace_files_sha256': {f.name: sha(f) for f in p.glob('exposure-rank*.jsonl')}, 'weights_shapes_equal_base_after_declared_tied_alias_restoration': True, 'omitted_tied_aliases': sorted(missing), 'sampled_parameter_update': sampled_update, 'independent_HF_CPU_load_and_forward': 'PASSED', 'weights_sha256': weights, 'continuous_optimizer_scheduler': True, 'on_disk_optimizer_state': False, 'inference_loadability': 'Independent CPU load/forward PASSED; formal vLLM inference follows'}
    dump(p / 'VALIDATED_MATH_SFT.json', report)
    reports.append(report)
    prior = got
assert reports[0]['weights_sha256'] != reports[1]['weights_sha256']
dump(out / 'TRAINING_DONE.json', {'status': 'COMPLETE', 'language': info['language'], 'epochs': reports, 'config_sha256': sha(a.config), 'data_manifest_sha256': sha(data / 'training_data.json')})
print(json.dumps(reports, indent=2))
