"""Read actual result configurations and physical-input traces, without rescoring."""
from pathlib import Path
import os
import argparse
import collections
import csv
import hashlib
import json

R = Path(__file__).resolve().parents[3]
L = Path(os.environ['P2_LOG_ROOT']) / 'evaluation'
OLD = Path(os.environ['REFERENCE_SW_LOG_ROOT'])
K = Path(os.environ['P2_CHECKPOINT_ROOT'])
BASE = str(Path(os.environ['QWEN3_4B']))
O = R / 'research/functional_representation_20261109/p2'

def load(p):
    return json.loads(p.read_text())

def only(root, pattern):
    xs = list(root.rglob(pattern));assert len(xs) == 1, (root, pattern, len(xs));return xs[0]

def lines(p):
    with p.open() as f:
        for line in f:yield json.loads(line)

def model_args(result):
    value = result['config']['model_args']
    return value if isinstance(value, dict) else dict(x.split('=', 1) for x in value.split(','))

def main():
    ap = argparse.ArgumentParser();ap.add_argument('--partial', action='store_true');args = ap.parse_args()
    records, pending, type_rows, trace_rows = [], [], [], []
    references = {}
    for model in ['Base', 'LegacyFull', 'all_layer', 'full_feature', 'ce_replay']:
        historical = model in ['Base', 'LegacyFull'];old = 'base' if model == 'Base' else 'lora'
        root = L / model
        adapter = None if model == 'Base' else str(K / (model + '_final_v1'))
        if model == 'LegacyFull':adapter = str(Path(os.environ['LEGACY_FULL_ADAPTER']))
        for stage in ['reading', 'core', 'summary', 'gmmlu']:
            if not historical and not (root / (stage + '.job/DONE.json')).exists():
                pending.append(model + '/' + stage);continue
            if stage == 'summary':
                run = OLD / (old + '-generation-v2/xlsum_sw_v2') if historical else root / 'summary'
                manifest = load(run / 'manifest.json');protocol = manifest['benchmark_protocol']
                assert manifest['scope'] == 'full' and manifest['batch_size'] == 8 and protocol['samples'] == 987
                assert protocol['max_input_tokens'] == 7168 and protocol['max_model_len'] == 8192
                assert protocol['enable_thinking'] is False
                assert protocol['decoding_config'] == dict(temperature=0, top_p=1, top_k=-1, max_new_tokens=256, stop_token_ids=[151645, 151643], seed=42)
                key = (manifest['prompt_tokens_sha256'], manifest['data_manifest_hash'], manifest['sample_ids_sha256'])
            else:
                if historical:
                    folder = old + ('-generation-v2/core/results' if stage == 'core' else '-harness-v1/' + ('belebele' if stage == 'reading' else 'gmmlu') + '/results')
                    source = OLD / folder
                else:source = root / ('hf/results' if stage == 'reading' else stage + '/results')
                result = load(only(source, 'results_*.json'));config = result['config'];actual = model_args(result)
                assert config['limit'] is None and actual['pretrained'] == actual['tokenizer'] == BASE
                assert actual['dtype'] == 'bfloat16' and str(actual['enable_thinking']).lower() == 'false'
                adapter_arg = 'lora_local_path' if stage == 'core' else 'peft'
                assert (actual.get(adapter_arg) or None) == adapter
                assert all(int(n) == (5 if stage == 'gmmlu' else 0) for n in result['n-shot'].values()) if stage != 'core' else True
                if stage in ['reading', 'gmmlu']:
                    assert int(config['batch_size']) == (4 if stage == 'reading' else 1)
                else:
                    assert result['n-shot'] == {'sft_v1_gsm8k': 5, 'sft_v1_ifeval': 0}
                    assert result['max_length'] == 8192
                    for name, task in result['configs'].items():
                        kw = task['generation_kwargs']
                        assert kw['max_gen_toks'] == 1280 and kw['temperature'] == 0 and kw['do_sample'] is False
                    assert str(actual['max_num_seqs']) == '8' and str(actual['max_num_batched_tokens']) == '8192'
                    trace = OLD / (old + '-generation-v2/core/completion-metadata.jsonl') if historical else root / 'core/completion-metadata.jsonl'
                    completion = list(lines(trace));assert len(completion) == 1860
                    assert max(x['generated_tokens'] for x in completion) <= 1280
                    physical = collections.Counter(x['prompt_token_sha256'] for x in completion)
                    if model == 'Base':references['physical_core_inputs'] = physical
                    assert physical == references['physical_core_inputs']
                    trace_rows.append(dict(model=model, actual_generations=len(completion),
                        input_tokens=sum(x['prompt_tokens'] for x in completion), generated_tokens=sum(x['generated_tokens'] for x in completion),
                        reached_cap=sum(x['generated_tokens'] == 1280 for x in completion),
                        normal_stop=sum(x['finish_reason'] == 'stop' for x in completion),
                        physical_input_multiset_matches_base=True))
                    counts = collections.Counter();correct = collections.Counter()
                    for x in lines(only(source, 'samples_sft_v1_ifeval_*.jsonl')):
                        for instruction, ok in zip(x['doc']['instruction_id_list'], x['inst_level_strict_acc']):
                            counts[instruction] += 1;correct[instruction] += int(ok)
                    assert sum(counts.values()) == 834
                    for kind in sorted(counts):
                        type_rows.append(dict(model=model, instruction_type=kind, instructions=counts[kind], correct=correct[kind],
                                              accuracy_percent=100 * correct[kind] / counts[kind], scope='DESCRIPTIVE_NOT_SELECTION'))
                key = (result.get('chat_template_sha'), json.dumps(result['n-shot'], sort_keys=True))
            if model == 'Base':references[stage] = key
            assert key == references[stage], (model, stage, 'Actual protocol differs from frozen Base')
            records.append(dict(model=model, stage=stage, actual_protocol_matches_base=True,
                                requested_adapter_matches_role=True, full_scope=True))
    complete = not pending
    if not args.partial:assert complete
    for name, rows in [('ifeval_instruction_types.csv', type_rows), ('core_generation_diagnostics.csv', trace_rows)]:
        with (O / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report = dict(complete=complete, stage_records_checked=len(records), expected_stage_records=20,
                  pending=pending, checks=records, official_scores_not_rescored=True,
                  description='Configurations and actual core token-input multiset match Base; type breakdown is descriptive only')
    (O / 'runtime_protocol_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'checks'}))

if __name__ == '__main__':
    main()
