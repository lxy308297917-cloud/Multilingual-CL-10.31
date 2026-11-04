# Decision stability pilot: frozen before model outputs

This is a diagnostic baseline comparison, not a proposed My Method.

## Data and readout

Uhura ARC-Easy official training split, source revision 561efa15460cba0965212b77f09bf6a7f57939a9. Shared semantic IDs EN/SW/HA, 400 IDs split 160 auxiliary /120 development /120 held out. Full protocol and exact hashes are in the immutable dataset protocol.json. It is project-fresh science QA, not a replacement for Belebele/MMLU. Existing pretraining exposure and translation equivalence cannot be certified.

P0: W1 SW FFT Final400 and existing sequential FFT100 on development only. Report original letter ACC, mean probability across four content rotations, complete answer likelihood (raw and token mean), and the 4x4 crossed label/position control. No best prompt selection. Keep every result, including unfavorable readouts. Teacher cache uses auxiliary EN/SW only. Joint correctness in both languages and unique top margin >1e-6 selects common IDs. Require at least16 such IDs to launch B/C/D; otherwise report inadequate reliable auxiliary pool without relaxing criteria. Native BF16 model candidate logits and FP32 four-candidate softmax for both teacher and student. P0/held-out scoring uses the previously validated FP32 export readout.

## Training

Same SW FFT400 initialization and HA6400 cached sequence as original sequential FFT. Seed42, two GPUs, microbatch1, accumulation8, effective16; AdamW1e-5, BF16,2048,no packing, assistant-only, non-thinking. Keep original400-step cosine schedule with20 warmup steps; stop at fixed100 (1600 HA sample exposures), not a shortened100-step schedule. A reuses checkpoint100 only after configuration and exposure identity verification. B/C/D each add one auxiliary semantic question per optimizer update, alternating EN and SW. Same reliable ID schedule and all four content rotations for each arm. Same lambda1, T1; no tuning. All use original HA supervision.

B: KL(teacher global per-language label marginal || mean current label probabilities across four rotations). C: KL(teacher per-question content-ID mean probability || current content-ID mean probability across rotations). D: mean supervised gold-label negative log probability on same four prompts. B deliberately tests marginal-only preservation; it does not constrain per-question decisions. Auxiliary computation is once per optimizer step, correctly normalized for actual HF gradient accumulation; rank copies are averaged by DDP. Preserve HA RNG around auxiliary forwards. Log HA exposure separately from auxiliary tokens/labels; auxiliary labels and source metadata are common to B/C/D even though B/C do not use gold CE. Teacher's offline cost is charged equally to B/C/D; A's lack of auxiliary and teacher compute is explicit, not called compute-matched.

Engineering probes use separate output directories, never count as100-step results. Exact resume and native optimizer states remain under original framework; do not alter shared LLaMA-Factory installation.

## Evaluation and decisions

After all fixed100 arms finish, evaluate W1 and A/B/C/D on the same development and held-out questions, with identical readouts and paired semantic-ID confidence intervals. No held-out winner-only reporting. Confirmatory fixed checkpoints also use existing frozen SW/EN Belebele, SW Global-MMLU/EN MMLU, HA reading/knowledge, translation, IFEval and GSM8K protocols; report completed and NOT_RUN entries faithfully. Existing exposed official benchmarks are supplemental fixed-checkpoint comparisons and never used to change lambda, steps, teacher filtering or choose a checkpoint. No English-degradation early stopping.

Compare HA learning and SW/EN retention together. Lower forgetting without meaningful HA learning is not a successful continual learner. No automatic400-step expansion. A future decision-level method requires content distillation to beat both label-marginal and same-data Replay at comparable HA learning; pilot evidence may remain inconclusive. No spectral intervention or new My Method in this stage. Distinguish statistical uncertainty, task mismatch, and optimization failure. Report actual main/auxiliary/teacher token and compute costs and all failures.

## Exact supplemental benchmark scope (fixed before these evaluations)

Eight job groups per100-step checkpoint: EN/SW Belebele (900 each,0-shot), HA Belebele (900,0-shot), EN MMLU/SW Global-MMLU/HA Global-MMLU (14,042 each,5-shot), locked core GSM8K/IFEval and SW MGSM/X-IFEval, and both full FLORES-200 devtest directions (1,012 each,3 dev shots). Translation uses the completed bench-v2.1 protocol, batch32 vLLM V1, rather than the older500-item HF subset. Reuse exactly matching W1 results where available; do not mix legacy and full translation rows. All official tasks were exposed in previous research and are supplementary fixed-budget reporting; only the new semantic-ID holdout is an untouched within-project holdout for this pilot. No formal score changes the already frozen training or teacher IDs.

## Replay implementation preflight clarification (before D started)

D uses full-vocabulary CE on the correct answer's final letter token for each of the same four prompts. The common chat whitespace continuation prefix is given as context for all B/C/D, and EOS is not added as an extra target. This is gold-answer-token SFT replay; it is not four-candidate-normalized CE. A candidate-only CE helper remains a tested reference and was not used for D. Both vocab head forward budgets already include the full vocab, so this correction does not add teacher or old input data. Separate two-step native Replay preflight precedes its100-step run. B/C running code and configurations remain frozen.
