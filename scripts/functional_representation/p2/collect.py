"""Frozen endpoint collection: raw per-item identities, no hyperparameter selection."""
from pathlib import Path
import os
import argparse
import collections
import csv
import hashlib
import json
import numpy as np

R = Path(__file__).resolve().parents[3]
L = Path(os.environ['P2_LOG_ROOT'])
OLD = Path(os.environ['REFERENCE_SW_LOG_ROOT'])
O = R / 'research/functional_representation_20261109/p2'
MODELS = ['Base', 'LegacyFull', 'all_layer', 'full_feature', 'ce_replay']
METRICS = [('Belebele target', 'ACC'), ('Belebele EN', 'ACC'), ('XL-Sum target', 'ROUGE-Lsum'),
           ('GSM8K', 'strict-match'), ('GSM8K', 'flexible-extract'),
           ('IFEval', 'prompt_level_strict_acc'), ('IFEval', 'inst_level_strict_acc'), ('Global-MMLU SW', 'ACC')]

def load(p):
    return json.loads(p.read_text())

def items(p):
    with p.open() as f:
        for line in f:
            yield json.loads(line)

def only(root, pattern):
    values = list(root.rglob(pattern))
    assert len(values) == 1, (root, pattern, len(values))
    return values[0]

def write_csv(path, rows):
    assert rows
    with path.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def signature(row):
    return tuple(row[x] for x in ['doc_hash', 'prompt_hash', 'target_hash'])

def read_mc(root, task, expected):
    result = load(only(root, 'results_*.json'))['results'][task]['acc,none']
    samples = {}
    confusion = np.zeros((4, 4), dtype=int)
    for x in items(only(root, 'samples_' + task + '_*.jsonl')):
        assert x['filter'] == 'none'
        values = [float(r[0][0]) for r in x['resps']]
        assert len(values) == 4 and all(np.isfinite(values))
        predicted = int(np.argmax(values))
        gold = int(x['doc']['correct_answer_num']) - 1 if task.startswith('belebele') else 'ABCD'.index(x['target'])
        correct = int(predicted == gold)
        assert correct == int(x['acc'])
        assert x['doc_id'] not in samples
        samples[x['doc_id']] = (signature(x), correct, 1)
        confusion[gold, predicted] += 1
    assert len(samples) == expected
    assert abs(sum(x[1] for x in samples.values()) / expected - result) < 1e-9
    return samples, confusion

def read_core(root):
    results = load(only(root, 'results_*.json'))['results']
    answer = {}
    records = {'strict-match': {}, 'flexible-extract': {}}
    response_hashes = {}
    for x in items(only(root, 'samples_sft_v1_gsm8k_*.jsonl')):
        filt = x['filter'];assert filt in records
        assert x['doc_id'] not in records[filt]
        records[filt][x['doc_id']] = (signature(x), float(x['exact_match']), 1)
        h = hashlib.sha256(json.dumps(x['resps'], sort_keys=True).encode()).hexdigest()
        if x['doc_id'] in response_hashes:
            assert h == response_hashes[x['doc_id']], 'Two filters must score the same generation'
        response_hashes[x['doc_id']] = h
    for filt, xs in records.items():
        assert len(xs) == 1319
        assert abs(sum(x[1] for x in xs.values()) / 1319 - results['sft_v1_gsm8k']['exact_match,' + filt]) < 1e-9
        answer[('GSM8K', filt)] = xs
    assert records['strict-match'].keys() == records['flexible-extract'].keys()
    prompt, instruction = {}, {}
    for x in items(only(root, 'samples_sft_v1_ifeval_*.jsonl')):
        flags = x['inst_level_strict_acc']
        assert len(flags) == len(x['doc']['instruction_id_list']) and flags
        assert bool(x['prompt_level_strict_acc']) == all(flags)
        assert x['doc_id'] not in prompt
        prompt[x['doc_id']] = (signature(x), int(x['prompt_level_strict_acc']), 1)
        instruction[x['doc_id']] = (signature(x), sum(flags), len(flags))
    assert len(prompt) == 541 and sum(x[2] for x in instruction.values()) == 834
    for metric, xs in [('prompt_level_strict_acc', prompt), ('inst_level_strict_acc', instruction)]:
        assert abs(sum(x[1] for x in xs.values()) / sum(x[2] for x in xs.values()) - results['sft_v1_ifeval'][metric + ',none']) < 1e-9
        answer[('IFEval', metric)] = xs
    strict, flex = records['strict-match'], records['flexible-extract']
    counts = collections.Counter((bool(strict[k][1]), bool(flex[k][1])) for k in strict)
    sensitivity = dict(both_correct=counts[(True, True)], strict_only=counts[(True, False)],
                       flexible_only=counts[(False, True)], neither_correct=counts[(False, False)])
    return answer, sensitivity

def read_summary(run, scores):
    manifest = load(run / 'manifest.json');aggregate = load(scores / 'summary.json')
    assert aggregate['samples'] == 987 and aggregate['run_status'] == 'VERIFIED_COMPLETE'
    assert hashlib.sha256((run / 'predictions.jsonl').read_bytes()).hexdigest() == aggregate['predictions_sha256']
    samples = {}
    for x in load(scores / 'rouge_per_sample.json'):
        assert x['id'] not in samples
        samples[x['id']] = ((manifest['prompt_tokens_sha256'], manifest['data_manifest_hash']), x['ROUGE-Lsum'] / 100, 1)
    assert len(samples) == 987 and set(samples) == set(manifest['sample_ids'])
    assert abs(100 * sum(x[1] for x in samples.values()) / 987 - aggregate['metrics']['ROUGE-Lsum']) < 1e-8
    return samples, aggregate

def pair_vectors(a, b):
    assert a.keys() == b.keys()
    keys = sorted(a)
    assert all(a[k][0] == b[k][0] and a[k][2] == b[k][2] for k in keys), 'Task/prompt/target identity changed'
    return np.array([a[k][1] for k in keys], float), np.array([b[k][1] for k in keys], float), np.array([a[k][2] for k in keys], float)

def interval(a, b, weights, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    bootstrap = []
    for _ in range(2000):
        idx = rng.integers(0, len(a), len(a))
        bootstrap.append(100 * (b[idx] - a[idx]).sum() / weights[idx].sum())
    return [float(v) for v in np.percentile(bootstrap, [2.5, 97.5])]

def main():
    ap = argparse.ArgumentParser();ap.add_argument('--partial', action='store_true');args = ap.parse_args()
    datasets, confusion, subjects, summaries, extraction = {}, [], [], [], []
    pending = []
    for model in MODELS:
        historical = model in ['Base', 'LegacyFull']
        old_prefix = 'base' if model == 'Base' else 'lora'
        root = L / 'evaluation' / model
        available = {stage: historical or (root / (stage + '.job/DONE.json')).exists() for stage in ['reading', 'core', 'summary', 'gmmlu']}
        for stage, ready in available.items():
            if not ready:pending.append(model + '/' + stage)
        if not args.partial and not all(available.values()):
            raise RuntimeError('Frozen formal stages still pending: ' + ','.join(pending))
        data = {}
        if available['reading']:
            hf = OLD / (old_prefix + '-harness-v1/belebele/results') if historical else root / 'hf/results'
            for language, task in [('target', 'belebele_swh_Latn_locked'), ('EN', 'belebele_eng_Latn_locked')]:
                value, matrix = read_mc(hf, task, 900)
                data[('Belebele ' + language, 'ACC')] = value
                confusion.append(dict(model=model, task='Belebele ' + language, samples=900,
                                      predicted_counts=matrix.sum(axis=0).tolist(), gold_counts=matrix.sum(axis=1).tolist(),
                                      confusion_gold_rows_pred_columns=matrix.tolist()))
        if available['core']:
            core = OLD / (old_prefix + '-generation-v2/core/results') if historical else root / 'core/results'
            values, sensitivity = read_core(core);data.update(values);extraction.append(dict(model=model, **sensitivity))
        if available['summary']:
            run = OLD / (old_prefix + '-generation-v2/xlsum_sw_v2') if historical else root / 'summary'
            scores = run.with_name(run.name + '-scores') if historical else root / 'summary-scores'
            values, aggregate = read_summary(run, scores);data[('XL-Sum target', 'ROUGE-Lsum')] = values
            for metric, value in aggregate['metrics'].items():
                summaries.append(dict(model=model, metric=metric, score=value, samples=987,
                                      primary=metric == 'ROUGE-Lsum', scope='SUMMARY_TEXT_NOT_TRANSLATION'))
        if available['gmmlu']:
            gm = OLD / (old_prefix + '-harness-v1/gmmlu/results') if historical else root / 'gmmlu/results'
            paths = list(gm.rglob('samples_global_mmlu_full_sw_*.jsonl'));assert len(paths) == 57
            values = {}
            for p in sorted(paths):
                task = p.name.removeprefix('samples_').rsplit('_', 1)[0]
                size = sum(1 for _ in p.open());xs, _ = read_mc(gm, task, size)
                for key, x in xs.items():values[(task, key)] = x
                subjects.append(dict(model=model, subject=task.removeprefix('global_mmlu_full_sw_').removesuffix('_locked'),
                                     samples=size, correct=sum(x[1] for x in xs.values()), accuracy_percent=100 * sum(x[1] for x in xs.values()) / size))
            assert len(values) == 14042;data[('Global-MMLU SW', 'ACC')] = values
        datasets[model] = data
    reference_rows = list(csv.DictReader((R / 'research/lora_layer_analysis_20261108/full_scores.csv').open()))
    absolute, paired, matrices = [], [], []
    for i, metric in enumerate(METRICS):
        present = [m for m in MODELS if metric in datasets[m]]
        base = datasets['Base'][metric]
        arrays = {}
        for model in present:
            a, b, weights = pair_vectors(base, datasets[model][metric]);arrays[model] = b
            score = 100 * b.sum() / weights.sum()
            if model in ['Base', 'LegacyFull']:
                label = 'Base' if model == 'Base' else 'Full'
                ref = next(x for x in reference_rows if x['language'] == 'sw' and x['model'] == label and (x['task'], x['metric']) == metric)
                assert abs(score - float(ref['score'])) < 1e-7
            absolute.append(dict(model=model, task=metric[0], metric=metric[1], score=score,
                                 unit='ROUGE points' if metric[1] == 'ROUGE-Lsum' else 'percent',
                                 samples_or_instructions=int(weights.sum()),
                                 source='historical same-protocol reference' if model in ['Base', 'LegacyFull'] else 'new fixed Final400',
                                 training_microbatch=1 if model == 'LegacyFull' else (0 if model == 'Base' else 2)))
        for model in ['all_layer', 'full_feature', 'ce_replay']:
            for reference in ['Base'] + (['all_layer'] if model != 'all_layer' else []) + (['ce_replay'] if model == 'full_feature' else []):
                if model not in arrays or reference not in arrays:continue
                a, b, weights = pair_vectors(datasets[reference][metric], datasets[model][metric])
                seed = [42, MODELS.index(model), MODELS.index(reference), i]
                lo, hi = interval(a, b, weights, seed)
                paired.append(dict(model=model, reference=reference, task=metric[0], metric=metric[1],
                                   difference=100 * (b - a).sum() / weights.sum(), ci95_low=lo, ci95_high=hi,
                                   bootstrap_resamples=2000, bootstrap_seed=json.dumps(seed),
                                   bootstrap_unit='prompt_cluster' if metric[1] == 'inst_level_strict_acc' else 'item',
                                   samples_or_instructions=int(weights.sum())))
        identity = hashlib.sha256(json.dumps([(str(k), *base[k][0]) for k in sorted(base)], ensure_ascii=False).encode()).hexdigest()
        matrices.append(dict(task=metric[0], metric=metric[1], models=present, identity_sha256=identity,
                             item_weights=weights.tolist(), values=np.stack([arrays[m] for m in present], axis=1).tolist()))
    write_csv(O / 'scores.csv', absolute);write_csv(O / 'summary_all_metrics.csv', summaries)
    write_csv(O / 'gmmlu_subject_scores.csv', subjects)
    if paired:write_csv(O / 'paired_changes.csv', paired)
    (O / 'numeric_items.json').write_text(json.dumps(dict(scope='Numeric per-item sufficient statistics only; no questions, answers, paths or weights',
          model_order=MODELS, bootstrap='SeedSequence([42,model_index,reference_index,metric_index])', metrics=matrices), separators=(',', ':')) + '\n')
    (O / 'reading_choice_audit.json').write_text(json.dumps(confusion, indent=2) + '\n')
    (O / 'gsm_extraction_sensitivity.json').write_text(json.dumps(extraction, indent=2) + '\n')
    report = dict(complete=not pending, pending=pending, absolute_rows=len(absolute), paired_rows=len(paired),
                  input_hashes_aligned=True, mc_argmax_independently_recomputed=True,
                  official_core_aggregates_match_raw_items=True, summary_aggregates_match_raw_items=True,
                  single_training_seed=True, multiple_correction=False, no_selection=True)
    (O / 'collection_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    if not pending:assert len(absolute) == 40 and len(paired) == 48 and len(subjects) == 285 and len(summaries) == 20
    print(json.dumps(report))

if __name__ == '__main__':
    main()
