"""Uniform old-ability monitoring, retaining the project's existing protocols.

EN Belebele900: original HF log-likelihood task, 0-shot. EN GSM8K1319 (strict
and flexible) and IFEval541/834: original sft_v1 vLLM definitions and decoding.
No early stopping and no model selection from any of these formal scores.
"""
from runtime import runtime_path
import argparse, collections, json, os, subprocess, sys, time, shutil
from pathlib import Path
from common import ROOT, MODEL, sha, dump

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--reuse-reading', type=Path)
    a = ap.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    if a.model != MODEL:
        v = json.loads((a.model / 'VALIDATED.json').read_text())
        assert v['full_epoch_exact_multiset']
    trainpy = runtime_path('train_python', '')
    evalpy = runtime_path('eval_python', '')
    weights = {p.name: sha(p) for p in a.model.glob('*.safetensors')}
    assert weights
    locked = Path(runtime_path('old_eval_data', ''))
    manifest = json.loads((locked / 'manifest.json').read_text())
    for f, h in manifest['file_sha256'].items():
        assert sha(locked / f) == h
    stack = json.loads((Path(runtime_path('retention_lock', '')) / '').read_text())
    for f, h in stack['tasks'].items():
        assert sha(Path(runtime_path('retention_core_tasks', '')) / '' / f) == h
    env = dict(os.environ, VLLM_USE_V1='0', VLLM_WORKER_MULTIPROC_METHOD='spawn', NLTK_DATA=runtime_path('old_nltk', ''))
    hf_args = f'pretrained={a.model},tokenizer={MODEL},dtype=bfloat16,enable_thinking=False'
    vv_args = f'pretrained={a.model},tokenizer={MODEL},dtype=bfloat16,max_model_len=8192,gpu_memory_utilization=0.85,max_num_seqs=32,max_num_batched_tokens=8192,enable_prefix_caching=True,seed=42,enforce_eager=True,enable_thinking=False'
    commands = [('en_reading', [str(Path(trainpy).parent / 'lm-eval'), 'run', '--model', 'hf', '--model_args', hf_args, '--tasks', 'belebele_eng_Latn_locked', '--include_path', str(Path(runtime_path('retention_reading_tasks', '')) / ''), '--num_fewshot', '0', '--apply_chat_template', '--batch_size', '4', '--device', 'cuda:0', '--output_path', str(a.output / 'en_reading/results'), '--log_samples', '--seed', '42']), ('en_core', [evalpy, str(ROOT / 'scripts/sft_v1/harness.py'), 'run', '--model', 'vllm', '--model_args', vv_args, '--tasks', 'sft_v1_gsm8k,sft_v1_ifeval', '--include_path', str(Path(runtime_path('retention_core_tasks', '')) / ''), '--apply_chat_template', '--batch_size', 'auto', '--output_path', str(a.output / 'en_core/results'), '--log_samples', '--seed', '42'])]
    dump(a.output / 'launch.json', dict(weights_sha256=weights, commands=commands, stack_sha256=sha(Path(runtime_path('retention_lock', '')) / ''), selection=False, source_sha256={str(p): sha(p) for p in [Path(__file__), ROOT / 'scripts/sft_v1/harness.py', ROOT / 'scripts/sft_v1/vllm_compat.py']}))
    start = time.monotonic()
    scores = {}
    audit = {}
    for name, cmd in commands:
        folder = a.output / name
        folder.mkdir()
        extra = {'SFT_VLLM_TRACE': str(folder / 'completion-metadata.jsonl')} if name == 'en_core' else {}
        if name == 'en_reading' and a.reuse_reading is not None:
            original = json.loads((a.reuse_reading / 'launch.json').read_text())
            assert original['weights_sha256'] == weights
            assert original['stack_sha256'] == sha(Path(runtime_path('retention_lock', '')) / '')
            oldcmd = original['commands'][0][1]
            newcmd = list(cmd)
            oi = oldcmd.index('--output_path') + 1
            ni = newcmd.index('--output_path') + 1
            oldcmd[oi] = newcmd[ni] = 'OUTPUT_RELOCATED_ONLY'
            assert oldcmd == newcmd
            source = a.reuse_reading / 'en_reading'
            assert len(list(source.glob('results/**/results_*.json'))) == 1
            copied = {}
            for f in source.rglob('*'):
                if not f.is_file():
                    continue
                rel = f.relative_to(source)
                dest = folder / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dest)
                assert sha(f) == sha(dest)
                copied[str(rel)] = sha(dest)
            dump(a.output / 'reading_reuse.json', {'source': str(source), 'same_weights_and_commands_verified': True, 'files_sha256': copied, 'inference_rerun': False})
        else:
            with (folder / 'eval.log').open('x') as log:
                subprocess.run(cmd, cwd=ROOT, env=dict(env, **extra), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)
        result_files = list(folder.glob('results/**/results_*.json'))
        assert len(result_files) == 1
        result = json.loads(result_files[0].read_text())
        scores[name] = result['results']
        for task, n in [('belebele_eng_Latn_locked', 900)] if name == 'en_reading' else [('sft_v1_gsm8k', 1319), ('sft_v1_ifeval', 541)]:
            files = list(result_files[0].parent.glob(f'samples_{task}_*.jsonl'))
            assert len(files) == 1
            groups = collections.defaultdict(list)
            for l in files[0].open():
                r = json.loads(l)
                groups[r['filter']].append(r)
            vals = {}
            for filt, rs in groups.items():
                assert len(rs) == len({r['doc_id'] for r in rs}) == n
                if name == 'en_reading':
                    assert all((float(r['acc']) in [0.0, 1.0] for r in rs))
                    count = sum((max(range(len(r['resps'])), key=lambda j: float(r['resps'][j][0][0])) == int(r['target']) for r in rs))
                    rate = count / n
                    assert abs(rate - result['results'][task]['acc,none']) < 1e-10
                    vals['acc,none'] = {'rate': rate, 'n': n}
                else:
                    for metric in rs[0]['metrics']:
                        flags = [x for r in rs for x in (r[metric] if isinstance(r[metric], list) else [r[metric]])]
                        assert all((x in [0, 1] for x in flags))
                        rate = sum(flags) / len(flags)
                        assert abs(rate - result['results'][task][metric + ',' + filt]) < 1e-10
                        vals[metric + ',' + filt] = {'rate': rate, 'n': len(flags)}
            audit[task] = vals
        print(json.dumps({'task': name, 'completed': True, 'seconds': time.monotonic() - start}), flush=True)
    dump(a.output / 'scores.json', scores)
    dump(a.output / 'accuracy_audit.json', audit)
    dump(a.output / 'DONE.json', dict(status='DONE', seconds=time.monotonic() - start, weights_sha256=weights, formal_test_used_for_selection=False))
if __name__ == '__main__':
    main()
