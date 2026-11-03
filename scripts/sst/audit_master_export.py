"""Read-only SST FP32-master versus BF16-forward/export geometry audit."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser()
    for key in ['project', 'base', 'checkpoint', 'initial', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--expected-step', type=int, required=True)
    p.add_argument('--export-dir', type=Path)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    sys.path.insert(0, str(a.project/'scripts/sst'))
    from projector import is_target
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    start = time.time()
    def dump(name, value):
        (a.output/name).write_text(json.dumps(value, indent=2)+'\n')
    def entries(folder):
        paths = list(folder.glob('*.safetensors'))
        result = {}
        for path in paths:
            with safe_open(path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    assert key not in result
                    result[key] = path
        return result
    def tensor(mapping, key):
        with safe_open(mapping[key], framework='pt', device='cpu') as f:
            return f.get_tensor(key)
    def energy(delta, basis):
        projected = basis.project(delta)
        return {'delta_squared_norm': float(delta.double().square().sum()),
                'free_squared_norm': float(projected.double().square().sum()),
                'outside_free_squared_norm': float((delta-projected).double().square().sum()),
                'changed_elements': int(torch.count_nonzero(delta))}
    try:
        saved = torch.load(a.checkpoint/'sst-state.pt', map_location='cpu', weights_only=False, mmap=True)
        assert saved['step'] == a.expected_step
        hashes = saved['master_sha256']; initial_sha = saved['initial_sha256']
        metadata = {'step': saved['step'], 'refresh_steps': saved['refresh_steps'], 'interval': saved['interval']}
        del saved
        initial = torch.load(a.initial, map_location='cpu', weights_only=False, mmap=True)
        masters = get_fp32_state_dict_from_zero_checkpoint(str(a.checkpoint), lazy_mode=True)
        bm, fm = entries(a.base), entries(a.export_dir or a.checkpoint)
        rows = []; all_export_matches = True
        for n in sorted(hashes):
            master = masters[n].contiguous()
            assert hashlib.sha256(master.detach().numpy().tobytes()).hexdigest() == hashes[n], n
            base = tensor(bm, n).float(); exported = tensor(fm, n)
            exact = torch.equal(master.to(exported.dtype), exported)
            all_export_matches &= exact
            row = {'name': n, 'elements': master.numel(), 'export_dtype': str(exported.dtype),
                   'export_equals_rounded_master': exact, 'target': is_target(n)}
            if is_target(n):
                assert torch.equal(base, initial['references'][n])
                b = initial['bases'][n]
                row['fp32_master'] = energy(master-base, b)
                row['bf16_export'] = energy(exported.float()-base, b)
                row['rounding_squared_norm'] = float((exported.float()-master).double().square().sum())
            else:
                row['fp32_master'] = {'delta_squared_norm': float((master-base).double().square().sum()),
                                        'changed_elements': int(torch.count_nonzero(master-base))}
                row['bf16_export'] = {'delta_squared_norm': float((exported.float()-base).double().square().sum()),
                                     'changed_elements': int(torch.count_nonzero(exported.float()-base))}
            rows.append(row)
            if len(rows) % 28 == 0:
                dump('progress.json', {'tensors': len(rows), 'total': len(hashes), 'seconds': time.time()-start})
        assert len(rows) == 310 and sum(x['target'] for x in rows) == 196
        totals = {}
        for representation in ['fp32_master', 'bf16_export']:
            totals[representation] = {group: {key: sum(row[representation].get(key, 0) for row in rows if row['target'] == target)
                for key in ['delta_squared_norm', 'free_squared_norm', 'outside_free_squared_norm', 'changed_elements']}
                for group, target in [('target', True), ('unconstrained', False)]}
            d = totals[representation]['target']
            d['free_energy_fraction'] = d['free_squared_norm']/max(d['delta_squared_norm'], 1e-60)
            d['outside_free_energy_fraction'] = d['outside_free_squared_norm']/max(d['delta_squared_norm'], 1e-60)
        dump('audit.json', {'source': str(a.checkpoint), 'export': str(a.export_dir or a.checkpoint), 'metadata': metadata,
             'initial_sha256_from_verified_source': initial_sha,
             'all310_master_hashes_verified': True, 'all_exports_equal_rounded_masters': all_export_matches,
             'totals': totals, 'rows': rows, 'seconds': time.time()-start,
             'interpretation': 'Fixed initial Base r99 geometry only. BF16 rounding can leave the master feasible space; this is not by itself a functional or benchmark effect.'})
        dump('DONE.json', {'status': 'COMPLETE', 'all_export_matches': all_export_matches,
                           'tensors': len(rows), 'seconds': time.time()-start})
    except BaseException as error:
        dump('FAILED.json', {'error': repr(error), 'seconds': time.time()-start})
        raise


if __name__ == '__main__':
    main()
