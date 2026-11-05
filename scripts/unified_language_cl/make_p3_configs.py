"""Regenerate the six frozen B-cohort configs; no training or scheduling."""
import argparse
import copy
import json
from pathlib import Path
import yaml
from common import ROOT, DATA, sha, oh, rows, dump
from make_configs import CKPT
from runtime import runtime_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', nargs='+', choices=['massive', 'xnli', 'mtop'], required=True)
    ap.add_argument('--reference-root', type=Path, default=Path(runtime_path('configs')))
    ap.add_argument('--output-root', type=Path, default=Path(runtime_path('configs')))
    args = ap.parse_args()
    for track in args.tracks:
        parent = DATA/('prepared-v2' if track == 'mtop' else 'prepared-v1')/track
        derived = DATA/'prepared-disjoint-v1'/track
        p = json.loads((parent/'protocol.json').read_text())
        b = json.loads((derived/'protocol.json').read_text())
        assert b['parent_protocol_sha256'] == sha(parent/'protocol.json')
        assert b['training_ids'] == p['disjoint_reserve_ids']
        assert len(b['training_ids']) == len(set(b['training_ids'])) == 3200
        assert not set(b['training_ids']) & set(p['training_ids'])
        assert b['steps'] == p['steps'] == 200
        analysis_name = 'P3_MTOP_SUPPLEMENT_v1.json' if track == 'mtop' else 'P3_ANALYSIS_PLAN_v1.json'
        source_analysis = ROOT/'configs/unified_language_cl'/analysis_name
        destination = Path(runtime_path('configs'))/analysis_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert sha(source_analysis) == sha(destination)
        else:
            with destination.open('xb') as stream:
                stream.write(source_analysis.read_bytes())
        for arm in ['single_th', 'seq_es_th']:
            reference = args.reference_root/track/f'{track}_{arm}_seed42_v1.yaml'
            original = yaml.safe_load(reference.read_text())
            config = copy.deepcopy(original)
            name = f'{track}_disjoint_{arm}_seed42_v1'
            config.update(dataset_dir=str(derived), tokenized_path=str(derived/'th_tokenized'),
                          output_dir=str(CKPT/name), logging_dir=str(Path(runtime_path('logs'))/name/'tensorboard'))
            permitted = {'dataset_dir', 'tokenized_path', 'output_dir', 'logging_dir'}
            assert {k:v for k,v in config.items() if k not in permitted} == {k:v for k,v in original.items() if k not in permitted}
            out = args.output_root/track/(name+'.yaml')
            assert not out.exists() and not out.with_suffix('.identity.json').exists()
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open('x') as stream:
                yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
            identity = json.loads(reference.with_suffix('.identity.json').read_text())
            identity.update(config_sha256=sha(out), protocol_sha256=sha(derived/'protocol.json'),
                            dataset_sha256=sha(derived/'th_train.jsonl'),
                            tokenized_files_sha256={str(f.relative_to(derived/'th_tokenized')):sha(f) for f in (derived/'th_tokenized').rglob('*') if f.is_file()},
                            common_input_semantic_order_sha256=oh([r['id'] for r in rows(derived/'th_train.jsonl')]),
                            actual_input_and_supervision=b['token_stats']['th:train'],
                            parent_config_sha256=sha(reference), P3_analysis_plan_sha256=sha(destination),
                            control_cohort='predeclared_B', status='FROZEN_NOT_LAUNCHED')
            dump(out.with_suffix('.identity.json'), identity)
            print(json.dumps({'track':track, 'arm':arm, 'config':str(out), 'training_started':False}))


if __name__ == '__main__':
    main()
