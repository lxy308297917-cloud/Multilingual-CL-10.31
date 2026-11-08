"""Posthoc descriptive update magnitude; never substitutes for task performance."""
from pathlib import Path
import os
import csv
import hashlib
import json
import re
import time
import torch
from safetensors.torch import load_file

R = Path(__file__).resolve().parents[3]
K = Path(os.environ['P2_CHECKPOINT_ROOT'])
O = R / 'research/functional_representation_20261109/p2'

def main():
    torch.set_num_threads(4);start = time.monotonic()
    reference = { (int(x['layer']), x['module']): x for x in csv.DictReader((R / 'research/lora_layer_analysis_20261108/module_norms.csv').open()) if x['language'] == 'sw'}
    paths = {'LegacyFull': Path(os.environ['LEGACY_FULL_ADAPTER'])}
    paths.update({m: K / (m + '_final_v1') for m in ['all_layer', 'full_feature', 'ce_replay']})
    rows, totals, hashes = [], {}, {}
    for model, root in paths.items():
        if model != 'LegacyFull':assert (root / 'TRAINING_DONE.json').exists()
        config = json.loads((root / 'adapter_config.json').read_text())
        assert config['r'] == 16 and config['lora_alpha'] == 32 and not config['fan_in_fan_out']
        assert not config['alpha_pattern'] and not config['rank_pattern'] and not config['use_rslora'] and not config['use_dora']
        tensors = load_file(str(root / 'adapter_model.safetensors'));squared = 0.
        for name, a in tensors.items():
            if '.lora_A.' not in name:continue
            b = tensors[name.replace('.lora_A.', '.lora_B.')]
            match = re.search(r'layers\.(\d+)\.(mlp|self_attn)\.(\w+)\.lora_A', name);assert match
            layer, kind, module = int(match[1]), match[2], match[3]
            aa, bb = a.double(), b.double()
            value = max(0., float(((aa @ aa.T) * (bb.T @ bb)).sum()) * 4) ** .5
            ref = reference[(layer, module)]
            if model == 'LegacyFull':assert abs(value - float(ref['delta_frobenius'])) < 1e-8
            squared += value ** 2
            rows.append(dict(model=model, layer=layer, kind=kind, module=module, effective_delta_frobenius=value,
                             base_frobenius=float(ref['base_frobenius']), relative_frobenius=value / float(ref['base_frobenius'])))
        assert sum(x['model'] == model for x in rows) == 252
        totals[model] = squared ** .5
        hashes[model] = hashlib.sha256((root / 'adapter_model.safetensors').read_bytes()).hexdigest()
    with (O / 'effective_update_norms.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report = dict(scope='POSTHOC descriptive magnitude audit, specified after first reading results; no model selection',
                  effective_delta_norms=totals, ratios_to_fresh_all_layer={k:v / totals['all_layer'] for k,v in totals.items()},
                  model_hashes=hashes, legacy_norms_match_existing_252_module_reference=True,
                  cpu_seconds=time.monotonic() - start, extra_gpu_jobs=0,
                  limitations=['Norm is not direction, functional importance or benchmark performance',
                    'No formal matched-norm uniform-scaling control was run for these new models',
                    'P1 uniform scaling matched Remove-Middle, not this new full-feature training result'])
    (O / 'effective_update_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'norms':totals,'ratios':report['ratios_to_fresh_all_layer'],'cpu_seconds':report['cpu_seconds']}))

if __name__ == '__main__':
    main()
