"""Finite tied-embedding restoration control, no training or saved model changes."""
import argparse, collections, hashlib, json, os, time
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ['base', 'model', 'protocol', 'registration', 'baseline_run', 'output']:
        p.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    a = p.parse_args(); assert not a.output.exists(); a.output.mkdir(parents=True); start = time.time()
    def dump(name, value): (a.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    def log(name, value):
        with (a.output/name).open('a') as f: f.write(json.dumps(value, ensure_ascii=False)+'\n')
    try:
        reg = json.loads(a.registration.read_text()); protocol = json.loads(a.protocol.read_text())
        assert reg['cohort_protocol_sha256'] == sha(a.protocol) and not reg['formal_test_used']
        assert reg['interventions'] == ['unchanged', 'restore_tied_embedding_alpha0.5',
            'restore_tied_embedding_alpha1.0', 'random_sign_matched_alpha0.5', 'random_sign_matched_alpha1.0']
        assert json.loads((a.model/'trainer_state.json').read_text())['global_step'] == 400
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        assert os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-') and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
        torch.manual_seed(42); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.device_count() == 1
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32,
                    attn_implementation='eager', local_files_only=True).to('cuda:0').eval()
        model.requires_grad_(False); model.config.use_cache = False
        embedding = model.get_input_embeddings().weight
        assert model.get_output_embeddings().weight.data_ptr() == embedding.data_ptr()
        original = embedding.detach().cpu().clone()
        old = None
        for path in a.base.glob('*.safetensors'):
            with safe_open(path, framework='pt', device='cpu') as f:
                if 'model.embed_tokens.weight' in f.keys():
                    assert old is None; old = f.get_tensor('model.embed_tokens.weight').float()
        assert old is not None and old.shape == original.shape
        delta = original-old
        generator = torch.Generator().manual_seed(reg['seed'])
        signs = torch.randint(0, 2, delta.shape, generator=generator, dtype=torch.int8)*2-1
        random = delta*signs; del signs
        norms = delta.double().square().sum(1).sqrt()
        assert torch.equal(random.abs(), delta.abs())
        provenance = {'registration_sha256': sha(a.registration), 'cohort_protocol_sha256': sha(a.protocol),
            'script_sha256': sha(Path(__file__)), 'model': str(a.model),
            'model_weights_sha256': {path.name: sha(path) for path in a.model.glob('*.safetensors')},
            'unique_parameter': 'model.embed_tokens.weight tied to lm_head.weight',
            'delta_squared_norm': float(norms.square().sum()), 'dtype': 'Saved export promoted toFP32/eager',
            'conditions': reg['interventions'], 'no_optimizer_or_model_write': True,
            'interpretation': 'Conditional post-training restoration; not independent input/output role identification or causal reconstruction of training'}
        dump('protocol.json', provenance)
        baseline_saved = json.loads((a.baseline_run/'DONE.json').read_text())
        assert baseline_saved['status'] == 'EXPLORATORY_DEV_COMPLETE'
        assert baseline_saved['predictions_sha256'] == sha(a.baseline_run/'predictions.jsonl')
        expected_baseline = {(x['semantic_id'], x['language']): x for x in
                             map(json.loads, (a.baseline_run/'predictions.jsonl').open())}
        letters = protocol['answer_token_ids']; summaries = {}; max_baseline_logit_difference = 0.
        @torch.inference_mode()
        def evaluate(label):
            nonlocal max_baseline_logit_difference
            counts, correct, loss = collections.Counter(), collections.Counter(), collections.Counter()
            for sid in protocol['confirmation_ids']:
                for lang in protocol['languages']:
                    row = protocol['encoded'][f'{sid}|{lang}|0']; gold = row['gold']
                    ids = torch.tensor([row['input_ids']], device='cuda:0')
                    scores = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   use_cache=False, logits_to_keep=1).logits[0, -1, letters]
                    assert torch.isfinite(scores).all(); pred = int(scores.argmax())
                    ce = float(torch.nn.functional.cross_entropy(scores[None], torch.tensor([gold], device='cuda:0')))
                    values = scores.cpu().tolist()
                    if label == 'unchanged':
                        previous = expected_baseline[sid, lang]
                        assert previous['predicted'] == 'ABCD'[pred] and previous['gold'] == 'ABCD'[gold]
                        difference = max(abs(x-y) for x,y in zip(values,previous['option_scores']))
                        max_baseline_logit_difference = max(max_baseline_logit_difference,difference)
                        assert difference < 1e-5, 'Same-model baseline equivalence failed'
                    counts[lang] += 1; correct[lang] += pred == gold; loss[lang] += ce
                    log('predictions.jsonl', {'condition': label, 'semantic_id': sid, 'language': lang,
                        'gold': 'ABCD'[gold], 'predicted': 'ABCD'[pred], 'correct': pred == gold,
                        'option_scores': values, 'four_way_ce_aux': ce})
            result = {lang: {'n': counts[lang], 'correct': correct[lang],
                      'acc': correct[lang]/counts[lang], 'four_way_ce_aux': loss[lang]/counts[lang]}
                      for lang in protocol['languages']}
            log('summaries.jsonl', {'condition': label, 'scores': result, 'seconds': time.time()-start})
            return result
        summaries['unchanged'] = evaluate('unchanged')
        for label in reg['interventions'][1:]:
            alpha = float(label.split('alpha')[-1]); direction = random if label.startswith('random') else delta
            value = original-alpha*direction
            actual_norms = (value-original).double().square().sum(1).sqrt()
            expected = alpha*norms; nonzero = expected > 0
            error = float(((actual_norms[nonzero]-expected[nonzero]).abs()/expected[nonzero]).max())
            assert error < 1e-4 and torch.count_nonzero(actual_norms[~nonzero]) == 0
            with torch.no_grad(): embedding.copy_(value.to(embedding.device))
            assert model.get_output_embeddings().weight.data_ptr() == embedding.data_ptr()
            log('interventions.jsonl', {'condition': label, 'alpha': alpha,
                'global_effective_delta_norm': float(actual_norms.square().sum().sqrt()),
                'max_row_relative_norm_error': error, 'tied_roles_preserved': True,
                'other_parameters_frozen': True})
            summaries[label] = evaluate(label)
            with torch.no_grad():
                embedding.copy_(original.to(embedding.device))
                assert torch.equal(embedding.cpu(), original)
        dump('DONE.json', {'status': 'EXPLORATORY_EMBEDDING_CONTROL_COMPLETE', 'results': summaries,
             'predictions_sha256': sha(a.output/'predictions.jsonl'), 'protocol_sha256': sha(a.output/'protocol.json'),
             'baseline_max_absolute_logit_difference': max_baseline_logit_difference,
             'seconds': time.time()-start, 'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
             'formal_test_used': False, 'adoption_claim_allowed': False})
    except BaseException as error:
        dump('FAILED.json', {'error': repr(error), 'seconds': time.time()-start}); raise


if __name__ == '__main__': main()
