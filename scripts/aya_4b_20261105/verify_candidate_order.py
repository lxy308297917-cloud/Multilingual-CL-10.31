"""Compare actual training input/label exposure order in two completed candidates."""
from profile_paths import resolve
import argparse, json, hashlib
from pathlib import Path
import yaml

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--second', type=Path, required=True)
    p.add_argument('--first-config', type=Path, required=True)
    p.add_argument('--second-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    configs = [yaml.safe_load(v.read_text()) for v in [a.first_config, a.second_config]]
    changes = {k for k in configs[0] if configs[0][k] != configs[1][k]}
    assert changes == {'learning_rate', 'output_dir', 'logging_dir'}
    fingerprints = []
    hashes = []
    for root in [a.first, a.second]:
        assert json.loads((root / 'TRAINING_DONE.json').read_text())['steps'] == 400
        source = root / 'exposure-rank0.jsonl'
        rows = [json.loads(v) for v in source.open()]
        assert len(rows) == 6400
        fingerprints.append([(v['update_step'], v['input_label_sha256'], v['input_tokens'], v['shifted_assistant_tokens']) for v in rows])
        hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())
    assert fingerprints[0] == fingerprints[1]
    result = {'status': 'VERIFIED_ALL6400_ORDERED_EXPOSURES_IDENTICAL', 'rows': 6400, 'steps': 400, 'scientific_config_difference': {'learning_rate': [v['learning_rate'] for v in configs]}, 'output_and_log_locations_independent': True, 'actual_exposure_sha256': hashes, 'formal_test_used_for_comparison': False}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
if __name__ == '__main__':
    main()
