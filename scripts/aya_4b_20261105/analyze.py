"""Audit actual complete results, export answer-free item scores, paired uncertainty.

A PARTIAL report never treats incomplete result files as completed benchmarks.
No scores or intervals affect training, data selection, or fixed Final selection.
"""
from profile_paths import resolve
import os, sys, argparse, json, csv, gzip, hashlib, math, collections, datetime, random
from pathlib import Path
import numpy as np
R = Path(resolve('ROOT'))
sys.path.insert(0, str(R / 'scripts/bench_v2'))
from paired_statistics import paired_corpus, paired_macro
from scoring import mt_metrics
LOG = Path(resolve('LOGS'))
ROLES = ['base', 'fft', 'lora', 'ssu']
BENCH_NAMES = ['flores200_en_sw_full_devtest_v2', 'flores200_sw_en_full_devtest_v2', 'xlsum_sw_v2', 'xlsum_en_v2']

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

def hs(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def rows(p):
    return [json.loads(s) for s in p.open() if s.strip()]

def one(paths):
    paths = list(paths)
    assert len(paths) == 1, [str(p) for p in paths]
    return paths[0]

def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')

def raw_response(x):
    return x['resps'][0][0]

def paired_ratio(items, base, repeats=1000):
    assert items.keys() == base.keys()
    keys = sorted(items)
    assert all((items[k]['identity'] == base[k]['identity'] and items[k]['denominator'] == base[k]['denominator'] for k in keys))
    grouped = collections.defaultdict(lambda: [0.0, 0.0])
    for k in keys:
        item, ref = (items[k], base[k])
        g = grouped[item['cluster']]
        g[0] += item['numerator'] - ref['numerator']
        g[1] += item['denominator']
    arr = np.array(list(grouped.values()))
    rng = np.random.default_rng(42)
    d = []
    for _ in range(repeats):
        sample = arr[rng.integers(0, len(arr), len(arr))]
        d.append(100 * sample[:, 0].sum() / sample[:, 1].sum())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {'delta': 100 * arr[:, 0].sum() / arr[:, 1].sum(), 'ci95_low': float(lo), 'ci95_high': float(hi), 'repeats': repeats, 'seed': 42, 'unit': 'paired source question; Belebele resamples shared passage clusters; IFEval instructions resample whole prompts', 'samples': len(keys), 'clusters': len(arr), 'multiple_comparisons': 'unadjusted exploratory intervals'}

class Collector:

    def __init__(self):
        self.scores = []
        self.items = {}
        self.identities = {}
        self.rawgen = {}
        self.status = []
        self.audits = []

    def add(self, role, task, metric, items, reported, unit='percent'):
        assert items
        value = 100 * sum((x['numerator'] for x in items.values())) / sum((x['denominator'] for x in items.values()))
        if unit == 'points':
            value /= 100
        assert abs(value - reported) < 1e-08, (role, task, metric, value, reported)
        self.scores.append({'model': role, 'task': task, 'metric': metric, 'score': value, 'unit': unit, 'samples': len(items), 'denominator': sum((x['denominator'] for x in items.values())), 'status': 'VERIFIED_COMPLETE'})
        self.items[role, task, metric] = items

    def mc(self, role, group):
        d = LOG / (role + '-harness-v1') / group
        if not (d / 'DONE.json').exists():
            self.status.append({'model': role, 'task_group': group, 'status': 'NOT_COMPLETE'})
            return
        f = one(d.glob('results/**/results_*.json'))
        summary = json.loads(f.read_text())
        samplefiles = sorted(f.parent.glob('samples_*.jsonl'))
        assert len(samplefiles) == (2 if group == 'belebele' else 57)
        combined = collections.defaultdict(dict)
        reported = {}
        for p in samplefiles:
            xs = rows(p)
            name = next((k for k in summary['results'] if p.name.startswith('samples_' + k + '_')))
            assert len(xs) == summary['results'][name]['sample_len']
            task = ('Belebele EN' if 'eng_Latn' in name else 'Belebele SW') if group == 'belebele' else 'MMLU EN' if group == 'mmlu' else 'Global-MMLU SW'
            local = 0
            for x in xs:
                assert x['filter'] == 'none' and len(x['resps']) == 4
                ll = [float(r[0][0]) for r in x['resps']]
                assert all((math.isfinite(v) for v in ll))
                gold = str(x['target']).strip()
                gold = 'ABCD'.index(gold) if gold in ['A', 'B', 'C', 'D'] else int(gold)
                assert gold in range(4)
                correct = int(max(range(4), key=lambda i: ll[i]) == gold)
                assert correct == int(x['acc'])
                local += correct
                key = name + '|' + str(x['doc_id'])
                assert key not in combined[task]
                identity = hs([x['doc_hash'], x['prompt_hash'], x['target_hash'], x['arguments']])
                cluster = hs(x['doc']['flores_passage']) if group == 'belebele' else key
                combined[task][key] = {'identity': identity, 'cluster': cluster, 'numerator': correct, 'denominator': 1, 'prediction_index': int(max(range(4), key=lambda i: ll[i]))}
            assert abs(local / len(xs) - summary['results'][name]['acc,none']) < 1e-10
            self.identities[str(p)] = sha(p)
            if group == 'belebele':
                reported[task] = summary['results'][name]['acc,none'] * 100
        for task, it in combined.items():
            expected = 900 if group == 'belebele' else 14042
            assert len(it) == expected
            if group != 'belebele':
                reported[task] = summary['results']['mmlu_locked' if group == 'mmlu' else 'global_mmlu_full_sw_locked']['acc,none'] * 100
            self.add(role, task, 'ACC', it, reported[task])
        self.audits.append({'model': role, 'task_group': group, 'candidate_likelihood_argmax_and_actual_gold_recomputed': True, 'sample_files': len(samplefiles), 'samples': sum((len(x) for x in combined.values()))})

    def core(self, role):
        d = LOG / (role + '-generation-v2')
        if not (d / 'DONE.json').exists():
            self.status.append({'model': role, 'task_group': 'GSM8K+IFEval', 'status': 'NOT_COMPLETE'})
            return
        f = one((d / 'core/results').glob('**/results_*.json'))
        result = json.loads(f.read_text())
        from lm_eval.filters.extraction import RegexFilter
        from lm_eval.api.metrics import exact_match_hf_evaluate
        from lm_eval.tasks.ifeval.utils import process_results
        ifeval_deterministic = 0
        ifeval_stochastic = []
        for task, n in [('sft_v1_gsm8k', 1319), ('sft_v1_ifeval', 541)]:
            p = one(f.parent.glob('samples_' + task + '_*.jsonl'))
            grouped = collections.defaultdict(list)
            for x in rows(p):
                grouped[x['filter']].append(x)
            cfg = result['configs'][task]
            for filt, xs in grouped.items():
                assert len(xs) == len({x['doc_id'] for x in xs}) == n
                allmetrics = collections.defaultdict(dict)
                for x in xs:
                    if task == 'sft_v1_gsm8k':
                        spec = next((y for y in cfg['filter_list'] if y['name'] == filt))
                        assert [y['function'] for y in spec['filter']] == ['regex', 'take_first']
                        kw = {k: v for k, v in spec['filter'][0].items() if k != 'function'}
                        parsed = RegexFilter(**kw).apply([[raw_response(x)]], [x['doc']])[0][0]
                        assert parsed == x['filtered_resps'][0]
                        metriccfg = next((y for y in cfg['metric_list'] if y['metric'] == 'exact_match'))
                        metric_kwargs = {k: v for k, v in metriccfg.items() if k in ['regexes_to_ignore', 'ignore_case', 'ignore_punctuation', 'ignore_numbers']}
                        hit = exact_match_hf_evaluate(predictions=[parsed], references=[x['target']], **metric_kwargs)['exact_match']
                        assert hit == x['exact_match']
                        metrics = {'exact_match': hit}
                    else:
                        state = random.getstate()
                        rescored = process_results(x['doc'], [raw_response(x)])
                        uses_random = random.getstate() != state
                        random.setstate(state)
                        uses_language_detector = any(('language' in instruction for instruction in x['doc']['instruction_id_list']))
                        metrics = {k: x[k] for k in ['prompt_level_strict_acc', 'inst_level_strict_acc', 'prompt_level_loose_acc', 'inst_level_loose_acc']}
                        assert metrics['prompt_level_strict_acc'] == all(metrics['inst_level_strict_acc']) and metrics['prompt_level_loose_acc'] == all(metrics['inst_level_loose_acc'])
                        assert len(metrics['inst_level_strict_acc']) == len(metrics['inst_level_loose_acc']) == len(x['doc']['instruction_id_list'])
                        if not uses_random and (not uses_language_detector):
                            for k, v in rescored.items():
                                assert v == x[k], (role, task, x['doc_id'], k)
                            ifeval_deterministic += 1
                        else:
                            ifeval_stochastic.append({'item_id_sha256': hs(str(x['doc_id'])), 'python_random_used': uses_random, 'language_detector': uses_language_detector})
                    for metric, value in metrics.items():
                        flags = value if isinstance(value, list) else [value]
                        assert flags and all((v in [0, 1] for v in flags))
                        key = str(x['doc_id'])
                        allmetrics[metric][key] = {'identity': hs([x['doc_hash'], x['prompt_hash'], x['target_hash'], x['arguments']]), 'cluster': key, 'numerator': int(sum(flags)), 'denominator': len(flags)}
                for metric, it in allmetrics.items():
                    label = 'GSM8K ' + filt if task == 'sft_v1_gsm8k' else 'IFEval'
                    metricname = 'ACC' if task == 'sft_v1_gsm8k' else metric
                    if task == 'sft_v1_ifeval' and metric.startswith('inst_'):
                        assert sum((y['denominator'] for y in it.values())) == 834
                    self.add(role, label, metricname, it, result['results'][task][metric + ',' + filt] * 100)
            self.identities[str(p)] = sha(p)
        self.audits.append({'model': role, 'task_group': 'core', 'official_gsm_regex_and_exact_match_reexecuted': True, 'ifeval_saved_official_flags_aggregated_and_instruction_counts_verified': True, 'ifeval_deterministic_prompts_reexecuted': ifeval_deterministic, 'ifeval_stochastic_or_language_prompts_not_claimed_exact_rescore': ifeval_stochastic, 'ifeval_note': 'Unchanged official scorer can sample fallback constraints for invalid parameters; primary scores always use the actual original Harness flags. Independent full deterministic replay is not claimed without captured scorer RNG state.', 'unique_gsm_questions': 1319, 'unique_ifeval_prompts': 541})

    def gen(self, role):
        root = LOG / (role + '-generation-v2')
        for name in BENCH_NAMES:
            d = root / name
            sc = root / (name + '-scores') / 'summary.json'
            if not sc.exists() or not (d / 'DONE.json').exists():
                self.status.append({'model': role, 'task_group': name, 'status': 'NOT_COMPLETE'})
                continue
            s = json.loads(sc.read_text())
            m = json.loads((d / 'manifest.json').read_text())
            assert s['run_status'] == 'VERIFIED_COMPLETE'
            p = d / 'predictions.jsonl'
            assert sha(p) == s['predictions_sha256']
            assert sha(d / 'manifest.json') == s['source_manifest_sha256']
            xs = rows(p)
            assert [x['id'] for x in xs] == m['sample_ids']
            assert len(xs) == (1012 if name.startswith('flores') else 987 if name == 'xlsum_sw_v2' else 11535)
            self.identities[str(p)] = sha(p)
            self.rawgen[role, name] = xs
            task = {'flores200_en_sw_full_devtest_v2': 'Translation EN->SW', 'flores200_sw_en_full_devtest_v2': 'Translation SW->EN', 'xlsum_sw_v2': 'Summary SW', 'xlsum_en_v2': 'Summary EN'}[name]
            if name.startswith('flores'):
                actual, _ = mt_metrics([x['prediction'] for x in xs], [x['reference'] for x in xs])
                for metric, value in actual.items():
                    assert abs(value - s['metrics'][metric]) < 1e-10
                    self.scores.append({'model': role, 'task': task, 'metric': metric, 'score': value, 'unit': 'points', 'samples': len(xs), 'denominator': len(xs), 'status': 'VERIFIED_COMPLETE'})
            else:
                per = json.loads((root / (name + '-scores') / 'rouge_per_sample.json').read_text())
                assert [x['id'] for x in per] == [x['id'] for x in xs]
                for metric in ['ROUGE-1', 'ROUGE-2', 'ROUGE-Lsum']:
                    it = {x['id']: {'identity': hs([x['source'], x['reference'], x['prompt_token_sha256']]), 'cluster': x['id'], 'numerator': float(v[metric]), 'denominator': 1} for x, v in zip(xs, per)}
                    self.add(role, task, metric, it, s['metrics'][metric], 'points')
            self.audits.append({'model': role, 'task_group': name, 'samples': len(xs), 'predictions_and_scoring_manifest_hashes_verified': True, 'actual_corpus_MT_recomputed': name.startswith('flores')})

    def export(self, out, complete):
        out.mkdir(parents=True, exist_ok=False)
        itemdir = out / 'per_item_scores'
        itemdir.mkdir()
        manifest = []
        for (role, task, metric), it in self.items.items():
            filename = role + '_' + task.replace(' ', '_').replace('->', '_') + '_' + metric.replace('/', '_') + '.csv.gz'
            p = itemdir / filename
            with gzip.open(p, 'wt', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['item_id_sha256', 'identity_sha256', 'cluster_sha256', 'numerator', 'denominator'])
                w.writeheader()
                for k, v in sorted(it.items()):
                    w.writerow({'item_id_sha256': hs(k), 'identity_sha256': v['identity'], 'cluster_sha256': hs(v['cluster']), 'numerator': v['numerator'], 'denominator': v['denominator']})
            manifest.append({'model': role, 'task': task, 'metric': metric, 'file': filename, 'rows': len(it), 'sha256': sha(p)})
        dump(itemdir / 'manifest.json', manifest)
        comparisons = []
        for (role, task, metric), it in self.items.items():
            if role == 'base' or ('base', task, metric) not in self.items:
                continue
            base = self.items['base', task, metric]
            c = paired_ratio(it, base)
            if task.startswith('Summary'):
                c = {**c, 'delta': c['delta'] / 100, 'ci95_low': c['ci95_low'] / 100, 'ci95_high': c['ci95_high'] / 100}
            comparisons.append({'model': role, 'task': task, 'metric': metric, **c})
        for (role, name), xs in self.rawgen.items():
            if role == 'base' or not name.startswith('flores') or ('base', name) not in self.rawgen:
                continue
            bx = self.rawgen['base', name]
            assert [x['id'] for x in xs] == [x['id'] for x in bx]
            assert all((x['reference'] == y['reference'] and x['source'] == y['source'] and (x['prompt_token_sha256'] == y['prompt_token_sha256']) for x, y in zip(xs, bx)))
            for metric in ['chrF++', 'BLEU']:
                comparisons.append({'model': role, 'task': 'Translation ' + ('EN->SW' if '_en_sw_' in name else 'SW->EN'), 'metric': metric, **paired_corpus([x['prediction'] for x in xs], [x['prediction'] for x in bx], [x['reference'] for x in xs], metric=metric)})
        from sacrebleu.metrics import CHRF, BLEU
        statsdir = out / 'corpus_statistics'
        statsdir.mkdir()
        stats_manifest = []
        for (role, name), xs in self.rawgen.items():
            if not name.startswith('flores'):
                continue
            task = 'Translation ' + ('EN->SW' if '_en_sw_' in name else 'SW->EN')
            for metric in ['chrF++', 'BLEU']:
                scorer = CHRF(char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False, eps_smoothing=False) if metric == 'chrF++' else BLEU(lowercase=False, tokenize='13a', smooth_method='exp', smooth_value=None, effective_order=False)
                stat = np.asarray(scorer._extract_corpus_statistics([x['prediction'] for x in xs], [[x['reference'] for x in xs]]), dtype=np.int64)
                point = scorer._compute_score_from_stats(stat.sum(axis=0).tolist()).score
                reported = next((x['score'] for x in self.scores if x['model'] == role and x['task'] == task and (x['metric'] == metric)))
                assert abs(point - reported) < 1e-10
                file = statsdir / (role + '_' + name + '_' + metric + '.csv.gz')
                with gzip.open(file, 'wt', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['item_id_sha256', 'identity_sha256', 'statistics'])
                    writer.writeheader()
                    for row, vector in zip(xs, stat):
                        writer.writerow({'item_id_sha256': hs(row['id']), 'identity_sha256': hs([row['source'], row['reference'], row['prompt_token_sha256']]), 'statistics': json.dumps(vector.tolist(), separators=(',', ':'))})
                stats_manifest.append({'model': role, 'task': task, 'metric': metric, 'file': file.name, 'rows': len(xs), 'statistics_dim': int(stat.shape[1]), 'sha256': sha(file), 'score': point, 'sacrebleu_version': '2.6.0', 'no_raw_text': True})
        dump(statsdir / 'manifest.json', stats_manifest)
        generation_diagnostics = []
        for (role, name), xs in self.rawgen.items():
            lengths = np.asarray([x['generated_tokens'] for x in xs])
            generation_diagnostics.append({'model': role, 'task_group': name, 'samples': len(xs), 'generation_hit_cap_count': sum((bool(x['generation_hit_cap']) for x in xs)), 'empty_prediction_count': sum((not x['prediction'].strip() for x in xs)), 'article_truncated_count': sum((bool(x['article_truncated']) for x in xs)), 'mean_generated_tokens': float(lengths.mean()), 'median_generated_tokens': float(np.median(lengths)), 'finish_reason_counts': dict(collections.Counter((x['finish_reason'] for x in xs))), 'use': 'Output-length/empty/truncation diagnosis only; no generation tuning or checkpoint selection'})
        dump(out / 'generation_diagnostics.json', generation_diagnostics)
        if complete:
            assert len(self.scores) == 80 and len(comparisons) == 60, (len(self.scores), len(comparisons))
        for filename, data in [('scores.json', self.scores), ('paired_comparisons.json', comparisons), ('source_identity.json', self.identities), ('score_audit.json', self.audits), ('status.json', self.status)]:
            dump(out / filename, data)
        for filename, data in [('scores.csv', self.scores), ('paired_comparisons.csv', comparisons)]:
            if data:
                keys = list(dict.fromkeys((k for x in data for k in x)))
                with (out / filename).open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(data)
        result = {'status': 'COMPLETE' if complete else 'PARTIAL_NOT_FINAL', 'score_rows': len(self.scores), 'paired_comparisons': len(comparisons), 'uncompleted_groups': self.status, 'created_at': datetime.datetime.now().astimezone().isoformat(), 'raw_text_excluded_from_per_item_export': True, 'intervals': 'Unadjusted exploratory paired95% intervals, conditional on seed42 models; not training-seed stability'}
        dump(out / 'DONE.json', result)
        print(json.dumps(result, ensure_ascii=False))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--partial', action='store_true')
    a = p.parse_args()
    c = Collector()
    for role in ROLES:
        for group in ['belebele', 'gmmlu', 'mmlu']:
            c.mc(role, group)
        c.gen(role)
        c.core(role)
    if not a.partial:
        assert not c.status, c.status
    c.export(a.output, not a.partial)
if __name__ == '__main__':
    main()
