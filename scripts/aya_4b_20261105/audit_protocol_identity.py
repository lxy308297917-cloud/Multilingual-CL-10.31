"""Require actual 4B artifacts and the same frozen scientific/execution protocol."""
from profile_paths import resolve
import argparse, json, hashlib
from pathlib import Path
R = Path(resolve('ROOT'))
LOG = Path(resolve('LOGS'))
CKPT = Path(resolve('CHECKPOINTS'))
BASE = Path(resolve('MODEL'))
CFG = R / 'configs/aya_4b_20261105'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def one(paths):
    ps = list(paths)
    assert len(ps) == 1, ps
    return ps[0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    audit = json.loads(a.training_audit.read_text())
    assert audit['status'] == 'VERIFIED_COMPLETE'
    lock = json.loads((CFG / 'model.lock.json').read_text())
    baseweights = {k: v['sha256'] for k, v in lock['files'].items() if k.endswith('.safetensors')}
    records = []
    generation_reference = {}
    baseconfig = json.loads((BASE / 'config.json').read_text())
    for role in ['base', 'fft', 'lora', 'ssu']:
        trained = CKPT / (role + ('_final_v1' if role == 'lora' else '_final_v2'))
        model = BASE if role in ['base', 'lora'] else trained
        if role in ['fft', 'ssu']:
            config = json.loads((model / 'config.json').read_text())
            for k in ['model_type', 'hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'vocab_size', 'tie_word_embeddings']:
                assert config[k] == baseconfig[k], (role, k)
            assert json.loads((model / 'generation_config.json').read_text())['eos_token_id'] == json.loads((BASE / 'generation_config.json').read_text())['eos_token_id']
        for group, batch, shots in [('belebele', '4', 0), ('gmmlu', '1', 5), ('mmlu', '4', 5), ('core', 'auto', None)]:
            root = LOG / (role + '-generation-v2') / 'core' if group == 'core' else LOG / (role + '-harness-v1') / group
            assert (root.parent / 'DONE.json').exists() if group == 'core' else (root / 'DONE.json').exists()
            f = one(root.glob('results/**/results_*.json'))
            d = json.loads(f.read_text())
            c = d['config']
            args = c['model_args']
            assert c['model'] == ('vllm' if group == 'core' else 'hf')
            assert str(c['batch_size']) == batch
            assert args['pretrained'] == str(model) and args['tokenizer'] == str(BASE)
            assert args['dtype'] == 'bfloat16' and args['enable_thinking'] is False
            for k in ['random_seed', 'numpy_seed', 'torch_seed', 'fewshot_seed']:
                assert c[k] == 42, (role, group, k)
            adapterkey = 'lora_local_path' if group == 'core' else 'peft'
            assert args.get(adapterkey) == (str(trained) if role == 'lora' else None)
            if group == 'core':
                for k, v in {'max_model_len': 8192, 'gpu_memory_utilization': 0.65, 'max_num_seqs': 8, 'max_num_batched_tokens': 8192, 'seed': 42, 'enforce_eager': True, 'enable_prefix_caching': True}.items():
                    assert args[k] == v, (role, k)
            for task, config in d['configs'].items():
                assert config['num_fewshot'] == (5 if group == 'core' and task == 'sft_v1_gsm8k' else 0 if group == 'core' else shots)
            records.append({'model': role, 'task_group': group, 'result_sha256': sha(f), 'actual_model_and_tokenizer_verified': True, 'fixed_batch_shots_seeds_verified': True})
        root = LOG / (role + '-generation-v2')
        assert (root / 'DONE.json').exists()
        expected = dict(baseweights) if role in ['base', 'lora'] else dict(audit['methods'][role]['actual_saved_weights_sha256'])
        if role == 'lora':
            expected.update(audit['methods']['lora']['actual_saved_weights_sha256'])
        for name in ['flores200_en_sw_full_devtest_v2', 'flores200_sw_en_full_devtest_v2', 'xlsum_sw_v2', 'xlsum_en_v2']:
            f = root / name / 'manifest.json'
            m = json.loads(f.read_text())
            assert m['execution_profile'] == 'aya4b-cuda-memory-v2'
            assert m['method'] == ('full' if role == 'fft' else role)
            assert m['model_revision'] == lock['revision']
            assert m['batch_size'] == 8
            assert m['scope'] == 'full'
            for k, v in {'max_model_len': 8192, 'max_num_seqs': 8, 'max_num_batched_tokens': 4096, 'gpu_memory_utilization': 0.65, 'dtype': 'bfloat16', 'generation_config': 'vllm'}.items():
                assert m['backend'][k] == v, (role, name, k)
            got = {Path(k).name: v for k, v in m['checkpoint_identity'].items() if k.endswith('.safetensors')}
            assert got == expected, (role, name)
            identity = {k: m[k] for k in ['benchmark_id', 'benchmark_protocol', 'data_manifest_hash', 'sample_ids', 'sample_ids_sha256', 'prompt_tokens_sha256', 'backend', 'batch_size']}
            if role == 'base':
                generation_reference[name] = identity
            else:
                assert identity == generation_reference[name], (role, name, 'protocol drift')
            records.append({'model': role, 'task_group': name, 'manifest_sha256': sha(f), 'actual_weight_hashes_verified': True, 'cross_model_prompt_data_and_execution_identity_verified': True})
    out = {'status': 'VERIFIED_ALL_FOUR_MODELS_FROZEN_PROTOCOL', 'groups_verified': len(records), 'records': records, 'official_model_revision': lock['revision'], 'canonical_generation_execution': 'v2; v1 excluded', 'physical_device_assertion': 'Additional recovery guard observes UUID only; no prompts or decoding modifications'}
    assert len(records) == 32
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'status': out['status'], 'groups_verified': len(records)}))
if __name__ == '__main__':
    main()
