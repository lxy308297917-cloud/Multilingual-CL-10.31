"""CPU verification of actual saved SSU mask protection; never edits weights."""
from profile_paths import resolve
import argparse, json, hashlib
from pathlib import Path
import torch
from safetensors import safe_open

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(x)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--trained', type=Path, required=True)
    p.add_argument('--mask', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    torch.set_num_threads(4)
    mask = json.loads(a.mask.read_text())
    assert mask['model_config_sha256'] == sha(a.base / 'config.json')

    def mapping(root):
        idx = root / 'model.safetensors.index.json'
        if idx.exists():
            return json.loads(idx.read_text())['weight_map']
        with safe_open(root / 'model.safetensors', framework='pt', device='cpu') as f:
            return {k: 'model.safetensors' for k in f.keys()}
    bm, tm = (mapping(a.base), mapping(a.trained))
    extra = set(tm) - set(bm)
    assert extra <= {'lm_head.weight'} and (not set(bm) - set(tm))
    assert json.loads((a.base / 'config.json').read_text())['tie_word_embeddings']
    assert json.loads((a.trained / 'config.json').read_text())['tie_word_embeddings']
    assert set(mask['parameter_masks']) <= set(bm)
    total = changed = protected = protected_changed = 0
    details = []
    handles = {}
    try:
        for root, mp in [(a.base, bm), (a.trained, tm)]:
            for filename in set(mp.values()):
                handles[str(root), filename] = safe_open(root / filename, framework='pt', device='cpu')
        if 'lm_head.weight' in extra:
            head = handles[str(a.trained), tm['lm_head.weight']].get_tensor('lm_head.weight')
            embed = handles[str(a.trained), tm['model.embed_tokens.weight']].get_tensor('model.embed_tokens.weight')
            assert torch.equal(head, embed), 'Serialized tied head differs from embedding'
            del head, embed
        for name in sorted(bm):
            b = handles[str(a.base), bm[name]].get_tensor(name)
            t = handles[str(a.trained), tm[name]].get_tensor(name)
            assert b.shape == t.shape
            nc = int(torch.count_nonzero(b != t))
            total += b.numel()
            changed += nc
            spec = mask['parameter_masks'].get(name)
            if spec:
                axis = {'columns': 1, 'rows': 0, 'elements': 0}[spec['kind']]
                assert tuple(spec['shape']) == tuple(b.shape)
                idx = torch.tensor(spec['indices'], dtype=torch.long)
                bp = b.index_select(axis, idx)
                tp = t.index_select(axis, idx)
                diff = int(torch.count_nonzero(bp != tp))
                protected += bp.numel()
                protected_changed += diff
                details.append({'parameter': name, 'protected_elements': bp.numel(), 'protected_changed': diff, 'all_changed': nc})
        assert total == 4022468096, total
        assert protected == 1816796928, protected
        assert protected_changed == 0, protected_changed
        assert changed > 0, 'Smoke must exercise actual nonzero updates'
        out = {'status': 'VERIFIED_ACTUAL_SAVED_MASK_PROTECTION', 'base_config_sha256': sha(a.base / 'config.json'), 'mask_sha256': sha(a.mask), 'total_unique_elements': total, 'protected_unique_elements': protected, 'protected_ratio': protected / total, 'protected_elements_changed': protected_changed, 'all_elements_changed': changed, 'unprotected_elements_changed': changed - protected_changed, 'serialized_extra_tied_head_verified_equal_and_not_double_counted': sorted(extra), 'parameters_checked': len(bm), 'masked_parameters_checked': len(details), 'details': details, 'base_weights_sha256': {f: sha(a.base / f) for f in sorted(set(bm.values()))}, 'trained_weights_sha256': {f: sha(a.trained / f) for f in sorted(set(tm.values()))}}
        a.output.write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps({k: v for k, v in out.items() if k not in ['details', 'base_weights_sha256', 'trained_weights_sha256']}, indent=2))
    finally:
        handles.clear()
if __name__ == '__main__':
    main()
