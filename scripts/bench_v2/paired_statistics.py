"""Paired uncertainty using pinned SacreBLEU corpus sufficient statistics."""
import importlib.metadata
import numpy as np
from sacrebleu.metrics import CHRF, BLEU


def paired_corpus(predictions, baseline, references, metric='chrF++', repeats=1000, seed=42):
    if importlib.metadata.version('sacrebleu') != '2.6.0':
        raise RuntimeError('Private sufficient-statistics API is locked to SacreBLEU 2.6.0')
    if not references or not len(predictions) == len(baseline) == len(references):
        raise ValueError('Nonempty aligned samples required')
    if repeats < 1:
        raise ValueError('Positive bootstrap repeats required')
    if metric == 'chrF++':
        scorer = CHRF(char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False, eps_smoothing=False)
    elif metric == 'BLEU':
        scorer = BLEU(lowercase=False, tokenize='13a', smooth_method='exp', smooth_value=None, effective_order=False)
    else:
        raise ValueError(metric)
    target = np.asarray(scorer._extract_corpus_statistics(predictions, [references]), dtype=np.int64)
    base = np.asarray(scorer._extract_corpus_statistics(baseline, [references]), dtype=np.int64)
    def score(stats):
        return scorer._compute_score_from_stats(stats.sum(axis=0).tolist()).score
    point = score(target) - score(base)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        indices = rng.integers(0, len(references), size=len(references))
        deltas.append(score(target[indices]) - score(base[indices]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return dict(delta=point, ci95_low=float(low), ci95_high=float(high), repeats=repeats,
                seed=seed, samples=len(references), unit='paired original sample IDs',
                algorithm='resample segment statistics then recompute corpus metric; not mean sentence scores',
                multiple_comparisons='unadjusted exploratory intervals')


def paired_macro(target, base, repeats=1000, seed=42):
    target, base = np.asarray(target, dtype=float), np.asarray(base, dtype=float)
    if target.ndim != 1 or target.shape != base.shape or not len(target) or repeats < 1:
        raise ValueError('Nonempty aligned score vectors and positive repeats required')
    differences = target - base
    if not np.isfinite(differences).all():
        raise ValueError('Nonfinite scores')
    rng = np.random.default_rng(seed)
    deltas = [float(differences[rng.integers(0, len(target), size=len(target))].mean()) for _ in range(repeats)]
    low, high = np.percentile(deltas, [2.5, 97.5])
    return dict(delta=float(differences.mean()), ci95_low=float(low), ci95_high=float(high),
                repeats=repeats, seed=seed, samples=len(target), unit='paired original article IDs',
                multiple_comparisons='unadjusted exploratory intervals')
