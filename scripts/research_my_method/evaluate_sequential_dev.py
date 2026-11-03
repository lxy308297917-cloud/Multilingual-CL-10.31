"""Same frozen MCQ development prompts for finished full-weight exports.

Promote exported weights to FP32 and use eager arithmetic, matching the local
coefficient pilot. This is not the historical BF16/few-shot formal benchmark.
For SST, comparing this model with the pilot's native FP32 master isolates
weight rounding while keeping the FP32 scoring implementation fixed.
"""
import argparse, collections, hashlib, json, os, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ['model', 'protocol', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    assert not a.output.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
    assert ',' not in os.environ['CUDA_VISIBLE_DEVICES']
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    a.output.mkdir(parents=True)
    start = time.time()
    def dump(name, value):
        (a.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    try:
        protocol = json.loads(a.protocol.read_text())
        assert protocol['version'] == 2 and not protocol['formal_test_used']
        if a.label != 'Base':
            trainer = json.loads((a.model/'trainer_state.json').read_text())
            assert trainer['global_step'] == trainer['max_steps'] == 400
        weights = sorted(a.model.glob('*.safetensors')); assert weights
        dump('protocol.json', {'label': a.label, 'model': str(a.model),
             'weights_sha256': {x.name: sha(x) for x in weights},
             'cohort_protocol_sha256': sha(a.protocol),
             'script_sha256': sha(Path(__file__)), 'semantic_ids': protocol['confirmation_ids'],
             'languages': protocol['languages'], 'arithmetic': 'FP32/eager/noTF32',
             'weights': 'Saved BF16 export promoted exactly to FP32',
             'prompt': 'Same pre-encoded0shot nonthinking tokens as coefficient pilot',
             'scoring': protocol['scoring'], 'formal_test_used': False,
             'historical_exposure': protocol['historical_exposure']})
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        assert torch.cuda.device_count() == 1
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32,
                    attn_implementation='eager', local_files_only=True).to('cuda:0').eval()
        model.config.use_cache = False
        torch.cuda.reset_peak_memory_stats()
        count, correct, losses = collections.Counter(), collections.Counter(), collections.Counter()
        letters = protocol['answer_token_ids']
        with torch.inference_mode(), (a.output/'predictions.jsonl').open('x') as out:
            for i, sid in enumerate(protocol['confirmation_ids']):
                for lang in protocol['languages']:
                    row = protocol['encoded'][f'{sid}|{lang}|0']; gold = row['gold']
                    ids = torch.tensor([row['input_ids']], device='cuda:0')
                    scores = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   use_cache=False, logits_to_keep=1).logits[0, -1, letters]
                    assert torch.isfinite(scores).all()
                    pred = int(scores.argmax())
                    loss = float(torch.nn.functional.cross_entropy(scores[None], torch.tensor([gold], device='cuda:0')))
                    count[lang] += 1; correct[lang] += pred == gold; losses[lang] += loss
                    out.write(json.dumps({'label': a.label, 'semantic_id': sid, 'language': lang,
                        'gold': 'ABCD'[gold], 'predicted': 'ABCD'[pred], 'correct': pred == gold,
                        'option_scores': scores.cpu().tolist(), 'four_way_ce_aux': loss})+'\n')
                if (i+1) % 20 == 0:
                    out.flush()
                    dump('progress.json', {'semantic_ids': i+1, 'total': len(protocol['confirmation_ids']), 'seconds': time.time()-start})
        summary = {lang: {'n': count[lang], 'correct': correct[lang], 'acc': correct[lang]/count[lang],
                         'four_way_ce_aux': losses[lang]/count[lang]} for lang in protocol['languages']}
        dump('DONE.json', {'status': 'EXPLORATORY_DEV_COMPLETE', 'label': a.label,
             'results': summary, 'seconds': time.time()-start,
             'predictions_sha256': sha(a.output/'predictions.jsonl'),
             'protocol_sha256': sha(a.output/'protocol.json'),
             'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
             'formal_test_used': False, 'adoption_claim_allowed': False})
    except BaseException as error:
        dump('FAILED.json', {'error': repr(error), 'seconds': time.time()-start}); raise


if __name__ == '__main__':
    main()
