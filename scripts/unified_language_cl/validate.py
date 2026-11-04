"""Check real exposure, frozen source data, exported weights and step counts."""
from runtime import runtime_path
import argparse, collections, json
from pathlib import Path
from common import ROOT, DATA, MODEL, sha, oh, dump, rows

def validate(config, steps=None):
    import yaml
    from datasets import load_from_disk
    from safetensors import safe_open
    cfg = yaml.safe_load(Path(config).read_text())
    out = Path(cfg['output_dir'])
    state = json.loads((out / 'trainer_state.json').read_text())
    expected = steps or cfg['max_steps']
    assert state['global_step'] == expected and state['max_steps'] == cfg['max_steps']
    data = load_from_disk(cfg['tokenized_path'])['train']
    known = collections.Counter((oh([x['input_ids'], x['labels']]) for x in data))
    counts = collections.Counter()
    sum_inputs = 0
    sum_targets = 0
    for rank in [0, 1]:
        p = out / f'exposure-rank{rank}.jsonl'
        trace = rows(p)
        account = json.loads((out / f'accounting-rank{rank}.json').read_text())
        assert len(trace) == expected * 8
        assert account['global_counts']['samples'] == expected * 16
        for r in trace:
            assert r['input_label_sha256'] in known
            counts[r['input_label_sha256']] += 1
            sum_inputs += r['input_tokens']
            sum_targets += r['shifted_assistant_tokens']
        assert account['trace_covers_all_local_samples']
    assert sum(counts.values()) == expected * 16
    if expected == cfg['max_steps']:
        assert counts == known, (len(counts), len(known))
    manifests = {}
    tensors = {}
    for p in sorted(out.glob('*.safetensors')):
        manifests[p.name] = sha(p)
        with safe_open(p, framework='pt', device='cpu') as f:
            for name in f.keys():
                assert name not in tensors
                t = f.get_slice(name)
                tensors[name] = {'shape': list(t.get_shape()), 'dtype': t.get_dtype()}
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    missing = set(index) - set(tensors)
    extra = set(tensors) - set(index)
    tied = json.loads((out / 'config.json').read_text()).get('tie_word_embeddings') is True
    assert not extra and (not missing or (missing == {'lm_head.weight'} and tied and ('model.embed_tokens.weight' in tensors))), (missing, extra)
    assert all((x['dtype'] == 'BF16' for x in tensors.values()))
    assert any(out.glob(f'checkpoint-{expected}/*/*optim_states.pt')) or (out / f'checkpoint-{expected}/optimizer.pt').exists(), 'Missing resumable optimizer state'
    result = dict(status='VALIDATED', steps=expected, schedule_steps=cfg['max_steps'], samples=expected * 16, unique_input_label_pairs=len(counts), input_tokens=sum_inputs, shifted_assistant_tokens=sum_targets, export_tensor_count=len(tensors), tied_alias_omitted=sorted(missing), weights=manifests, export_dtypes=sorted({t['dtype'] for t in tensors.values()}), config_sha256=sha(config), exposure_hashes={r: sha(out / f'exposure-rank{r}.jsonl') for r in [0, 1]}, checkpoint_optimizer_preserved=True, full_epoch_exact_multiset=expected == cfg['max_steps'])
    dump(out / 'VALIDATED.json', result)
    print(json.dumps(result), flush=True)
    return result
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--steps', type=int)
    a = ap.parse_args()
    validate(a.config, a.steps)
