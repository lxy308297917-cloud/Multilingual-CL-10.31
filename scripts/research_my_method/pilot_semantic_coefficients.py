"""Finite, norm-matched interventions in acquired SST update coordinates.

No optimizer, checkpoint write, formal test, or adoption decision. Engineering
mode reads only the fit partition. Full mode requires a 400-step source model.
"""
import argparse, collections, hashlib, json, math, os, sys, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from coefficient_geometry import centered_cross_space, project_coefficient


class CoefficientLinear(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        self.original = original
        self.register_buffer('u', mode['u'].detach())
        self.register_buffer('v', mode['v'].detach())
        self.coefficient = nn.Parameter(torch.zeros(len(mode['s'])))

    def forward(self, x):
        return self.original(x) + ((x @ self.v) * self.coefficient) @ self.u.T


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ['base', 'checkpoint', 'bank', 'protocol', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--engineering-only', action='store_true')
    p.add_argument('--training-record', type=Path)
    p.add_argument('--sst-code', type=Path, default=Path(__file__).resolve().parents[1]/'sst')
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    start = time.time()

    def dump(name, value):
        (a.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

    def log(name, value):
        with (a.output/name).open('a') as f:
            f.write(json.dumps(value, ensure_ascii=False)+'\n')

    try:
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        protocol = json.loads(a.protocol.read_text())
        assert protocol['version'] == 2 and not protocol['formal_test_used']
        # Refreshed native states contain our own serialized Subspace objects.
        sys.path.insert(0, str(a.sst_code))
        source = torch.load(a.checkpoint/'sst-state.pt', map_location='cpu',
                            weights_only=False, mmap=True)
        assert a.engineering_only or source['step'] == 400
        if not a.engineering_only:
            assert a.training_record is not None
            receipt = json.loads((a.training_record/'COMPLETED.json').read_text())
            assert receipt['exit_code'] == 0 and receipt['global_step'] == 400
            assert Path(receipt['output']).resolve() == a.checkpoint.parent.resolve()
            assert a.checkpoint.name == 'checkpoint-400'
        bank_manifest = json.loads((a.bank.parent/'manifest.json').read_text())
        assert sha(a.bank) == bank_manifest['bank_sha256']
        assert Path(bank_manifest['source_checkpoint']).resolve() == a.checkpoint.resolve()
        bank = torch.load(a.bank, map_location='cpu', weights_only=False)
        assert bank['source_checkpoint_metadata']['step'] == source['step']
        assert bank['initial_sha256'] == source['initial_sha256']
        modes = bank['modes']
        assert len(modes) == 196
        fit = protocol['engineering_fit_ids'] if a.engineering_only else protocol['fit_ids']
        evaluate_ids = protocol['engineering_eval_ids'] if a.engineering_only else protocol['confirmation_ids']
        assert not set(fit) & set(evaluate_ids)
        if a.engineering_only:
            assert not set(evaluate_ids) & set(protocol['confirmation_ids'])
        frozen = {k: v for k, v in protocol.items() if k != 'encoded'}
        frozen.update(engineering_only=a.engineering_only, actual_fit_ids=fit,
                      actual_evaluation_ids=evaluate_ids, source_step=source['step'],
                      source_checkpoint=str(a.checkpoint), protocol_sha256=sha(a.protocol),
                      bank_sha256=bank_manifest['bank_sha256'],
                      script_sha256=sha(Path(__file__)))
        dump('protocol.json', frozen)

        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        masters = get_fp32_state_dict_from_zero_checkpoint(str(a.checkpoint), lazy_mode=True)
        cfg = AutoConfig.from_pretrained(a.base, local_files_only=True)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32,
                                               attn_implementation='eager')
        original = dict(model.named_parameters())
        assert set(original) == set(source['master_sha256'])
        with torch.no_grad():
            for n, param in original.items():
                value = masters[n].contiguous()
                assert tensor_sha(value) == source['master_sha256'][n], n
                param.copy_(value)
                param.requires_grad_(False)
        dump('master-verification.json', {'count': len(original),
             'values': sum(x.numel() for x in original.values()), 'all_sha256_match': True})
        del masters, source, original
        wrappers = {}
        for name in sorted(modes):
            module_name = name[:-len('.weight')]
            parent_name, child = module_name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            linear = getattr(parent, child)
            assert isinstance(linear, nn.Linear)
            wrapper = CoefficientLinear(linear, modes[name])
            setattr(parent, child, wrapper)
            wrappers[name] = wrapper
        model.to('cuda:0')
        model.config.use_cache = False
        assert sum(x.numel() for x in model.parameters() if x.requires_grad) == bank['coefficient_count']
        assert not any(isinstance(x, nn.Dropout) and x.p > 0 for x in model.modules())
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.cuda.reset_peak_memory_stats()
        letters = protocol['answer_token_ids']
        encoded = protocol['encoded']

        def scores(sid, lang, rotation=0):
            row = encoded[f'{sid}|{lang}|{rotation}']
            ids = torch.tensor([row['input_ids']], device='cuda:0')
            values = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=False, logits_to_keep=1).logits[0, -1, letters]
            assert torch.isfinite(values).all()
            return values, row['gold']

        gradients = {}
        for lang in protocol['languages']:
            per_name = {n: [] for n in wrappers}
            model.train()
            for i, sid in enumerate(fit):
                model.zero_grad(set_to_none=True)
                for rotation in protocol['option_rotations']:
                    values, gold = scores(sid, lang, rotation)
                    loss = F.cross_entropy(values.unsqueeze(0), torch.tensor([gold], device='cuda:0'))/4
                    loss.backward()
                for n, module in wrappers.items():
                    grad = module.coefficient.grad
                    assert grad is not None and torch.isfinite(grad).all(), n
                    per_name[n].append(grad.detach().cpu().clone())
                log('progress.jsonl', {'phase': 'fit_gradient', 'language': lang,
                    'item': i+1, 'total': len(fit), 'seconds': time.time()-start})
            gradients[lang] = {n: torch.stack(v) for n, v in per_name.items()}
        torch.save(gradients, a.output/'fit-coefficient-gradients.pt')
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing_disable()
        model.eval()
        categories = [protocol['categories'][sid] for sid in fit]
        permutation = list(range(len(fit)))
        for category in sorted(set(categories)):
            indices = [i for i, c in enumerate(categories) if c == category]
            assert len(indices) >= 2
            for i, j in enumerate(indices):
                permutation[j] = indices[(i+1) % len(indices)]
        assert all(i != j for i, j in enumerate(permutation))
        active, directions, rows = [], {}, []
        for n in wrappers:
            en, sw = gradients['en'][n], gradients['sw'][n]
            paired, eigenvalues = centered_cross_space(en, sw, categories)
            rank = paired.shape[1]
            shuffled, _ = centered_cross_space(en, sw[permutation], categories, rank=rank)
            generator = torch.Generator().manual_seed(int(hashlib.sha256(
                ('coefficient-random-v2|'+n).encode()).hexdigest()[:12], 16))
            random = torch.linalg.qr(torch.randn(en.shape[1], rank, generator=generator), mode='reduced').Q
            raw = -gradients['ha'][n].mean(0)
            ds = {'raw': raw}
            for key, q in [('shared', paired), ('shuffled', shuffled), ('random', random)]:
                ds['block_'+key] = project_coefficient(raw, q)
                ds['reuse_'+key] = project_coefficient(raw, q, reuse=True)
            rn = float(raw.double().norm())
            valid = 0 < rank < len(raw) and rn > 0 and all(
                float(x.double().norm()) > rn*1e-8 for x in ds.values())
            if valid:
                active.append(n)
                directions[n] = {key: x*(rn/float(x.double().norm())) for key, x in ds.items()}
            rows.append({'name': n, 'dimension': len(raw), 'rank': rank,
                         'active': valid, 'raw_norm': rn, 'paired_eigenvalues': eigenvalues})
        dump('fit-only-active-subset.json', {'rows': rows, 'active_matrices': len(active),
             'shuffled_permutation': permutation, 'frozen_before_evaluation': True})
        assert active, 'No comparable directions; report infeasible, do not invent noise'
        old_squared = sum(float(modes[n]['s'].double().square().sum()) for n in active)
        raw_squared = sum(float(directions[n]['raw'].double().square().sum()) for n in active)
        assert old_squared > 0 and raw_squared > 0
        dump('coverage.json', {'active_matrices': len(active), 'all_matrices': 196,
             'active_old_update_squared_norm': old_squared,
             'all_old_update_squared_norm': bank['projected_update_squared_norm'],
             'active_old_energy_fraction': old_squared/bank['projected_update_squared_norm'],
             'coefficients': sum(len(modes[n]['s']) for n in active)})

        @torch.no_grad()
        def evaluate(label):
            count, correct, losses = collections.Counter(), collections.Counter(), collections.Counter()
            for sid in evaluate_ids:
                for lang in protocol['languages']:
                    values, gold = scores(sid, lang)
                    pred = int(values.argmax())
                    ce = float(F.cross_entropy(values.unsqueeze(0), torch.tensor([gold], device='cuda:0')))
                    count[lang] += 1
                    correct[lang] += pred == gold
                    losses[lang] += ce
                    log('predictions.jsonl', {'condition': label, 'semantic_id': sid,
                        'language': lang, 'predicted': 'ABCD'[pred], 'gold': 'ABCD'[gold],
                        'correct': pred == gold, 'option_scores': values.cpu().tolist(),
                        'four_way_ce_aux': ce})
            summary = {lang: {'n': count[lang], 'correct': correct[lang],
                       'acc': correct[lang]/count[lang], 'four_way_ce_aux': losses[lang]/count[lang]}
                       for lang in protocol['languages']}
            log('summaries.jsonl', {'condition': label, 'scores': summary,
                                  'seconds': time.time()-start})
            return summary

        results = {'W1': evaluate('W1')}
        fractions = [.1] if a.engineering_only else protocol['norm_fractions']
        conditions = protocol['conditions']
        for fraction in fractions:
            target_norm = fraction*math.sqrt(old_squared)
            for condition in conditions:
                measured_squared = 0.
                worst_error = 0.
                with torch.no_grad():
                    for n in active:
                        value = directions[n][condition]*(target_norm/math.sqrt(raw_squared))
                        wrappers[n].coefficient.copy_(value.to('cuda:0'))
                        # Exact effective-delta norm via Frobenius Gram; no dense allocation.
                        u, v = modes[n]['u'].double(), modes[n]['v'].double()
                        gram = (u.T@u)*(v.T@v)
                        x = value.double()
                        norm_squared = float(x@gram@x)
                        expected = target_norm*float(directions[n]['raw'].double().norm())/math.sqrt(raw_squared)
                        worst_error = max(worst_error, abs(math.sqrt(norm_squared)-expected)/expected)
                        measured_squared += norm_squared
                assert worst_error < 1e-4
                assert abs(math.sqrt(measured_squared)-target_norm)/target_norm < 1e-4
                label = f'{condition}-fraction{fraction}'
                log('interventions.jsonl', {'condition': label, 'target_effective_weight_norm': target_norm,
                     'measured_effective_weight_norm': math.sqrt(measured_squared),
                     'max_per_matrix_relative_norm_error': worst_error,
                     'all_other_weights_frozen': True})
                results[label] = evaluate(label)
                with torch.no_grad():
                    for module in wrappers.values():
                        module.coefficient.zero_()
                        assert int(torch.count_nonzero(module.coefficient)) == 0
        dump('DONE.json', {'status': 'ENGINEERING_COMPLETE' if a.engineering_only else 'EXPLORATORY_COMPLETE',
             'results': results, 'elapsed_seconds': time.time()-start,
             'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
             'formal_test_used': False, 'adoption_claim_allowed': False,
             'protocol_sha256': sha(a.protocol), 'bank_sha256': bank_manifest['bank_sha256']})
        print('DONE '+str(a.output), flush=True)
    except BaseException as error:
        dump('FAILED.json', {'error': repr(error), 'seconds': time.time()-start})
        raise


if __name__ == '__main__':
    main()
