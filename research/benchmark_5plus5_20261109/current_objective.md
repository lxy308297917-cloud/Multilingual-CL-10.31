# Benchmark extension: five target + five general

User authorization 2026-11-09: add SIB-200 SW/AR/RU and missing historical AR/RU Global-MMLU; total account GPUs temporarily at most5 for this extension. Other frozen queues keep their existing4 cap. No global GPU defaults changed. No new training or seeds for this supplement; existing formal training continues.

Target5: Belebele ACC, XNLI ACC, XL-Sum ROUGE-Lsum, Global-MMLU ACC, SIB-200 length-normalized likelihood ACC. General5: GSM8K flexible, official IFEval prompt strict, HumanEval+, EN MMLU, EN Belebele. Preserve EN XNLI, GSM strict, IFE instruction strict and translation as supplemental results.

Reuse all compatible completed scores. Distinguish historical micro1 LoRAs, aligned micro2 LoRAs, original36 FullRep and common72 family. TargetACC4 averages the four ACCs only; missing cells produce no average. XL-Sum remains separate. Cross-language macro averages require identical task/protocol sets, with per-language Base deltas.

SIB200 official204 test per language, fixed English labels/prompt, zero-shot, Harness multiple_choice; acc_norm primary (official Harness character-length normalization), raw acc secondary, no test-driven choice. Report macroF1, confusion, per-class recall, predicted distribution, majority baseline, paired intervals, FLORES/Belebele and training overlap. No claim that Belebele and SIB use independent content.

Deliver complete historical baseline and SW method tables, extend existing formal endpoints with SIB when ready without retraining, raw scores/identities,57-subject GM detail, costs and GitHub handoff.

## Latest user priority rule,22:32

Allow four already-running trainings to finish naturally; pause five queued training coordinators before launch (ARfull/ARkd/ARrandom/RUkd/matchedFFT). Formal5+5 and XQuAD supplement queues are the only new GPU workloads; no holding GPUs or replacement training. Mechanism diagnostics and extra translation stages are deferred. Keep USER_DEFERRED rows visible, not failures or zeros. Main5+5 scientific evaluator scripts remain frozen; only waiting CPU benchmark attempts were archived/requeued under a filtered scheduler. Account cap remains temporarily5 as explicitly authorized for this supplement.

## Complete General5 for historical rows

Reuse existing GSM8K,IFEval,EN Belebele and SW EN MMLU. Only fill three missing historical-LoRA HumanEval+ runs (SW/AR/RU) and two missing AR/RU historical-LoRA full EN MMLU runs. Same original frozen code generation/isolated EvalPlus execution and5-shot likelihood protocol; no extra training or scoring retuning. These five missing General5 tasks belong to the prioritized benchmark supplement, not new algorithm experiments.
