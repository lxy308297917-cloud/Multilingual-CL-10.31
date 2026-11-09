# Current authorized objective

Complete SW/AR/RU five-method fixed-Final400 formal comparisons from the corrected latest ChatGPT plan. This supersedes the previous conditional P3 goal and mistaken crosslayer solver family. No performance gate, no new seeds, no formal-test-derived weights.

## Necessary protocol alignment

Existing SW ordinary LoRA micro2/GA8 is reusable. AR/RU old ordinary LoRA micro1/GA16 are historical calibration adapters and will receive aligned micro2/GA8 control reruns.

Old uniform representation uses 36 block endpoints, which cannot distinguish Attention/MLP weights. New common feature set has 72 sites: post-attention residual (input to post_attention_layernorm), and post-MLP residual (decoder block output), at the same protected answer positions on the exact same EN build128. Six anatomical groups each contain 12 sites. All uniform/random/function runs use these same features; uniform weights1 is the exact new objective baseline. The old SW36-site FullRep result remains separately labeled and is not passed off as the new common-feature control. This requires 14 new runs, rather than the provisional11.

R_g = mean normalized squared feature drift over the12 sites in group g, divided by the fixed capability scale; total preservation = (1/6) sum_g w_g R_g. All six weights average1, bounded[0.5,2]. No layers frozen. Same target order, optimizer, old example order, teacher Base and scalar coefficient1.

Function weights: fixed attenuation0.75 and matched-total-effective-update-norm uniform controls. On build-only EN math/instruction and target reading/summary CE, risk = mean(old relative recovery compared with the matched uniform control) - max(0, mean(target relative cost compared with matched uniform)). Relative differences divide each reference NLL by max(NLL,0.1). Let z=max(risk,0); if z.sum=0 all weights1, otherwise w = 0.5 + 3*z/z.sum, then Euclidean project onto sum6 and [0.5,2]. Same rule across languages; calibration is part of algorithm, not an effectiveness screen. Random control uses NumPy default_rng(61415).permutation(6), same multiset. Freeze before new formal tests.

The reference calibration adapters are the original per-language completed LoRAs; their microbatch differs from new training but no new formal outcome is used to fit the weights. Both legacy and aligned ordinary baselines will be labeled explicitly.

Formal target: Belebele, XL-Sum, official Global-MMLU and XNLI where available. EN: GSM8K, official IFEval strict, MMLU, HumanEval+, Belebele, XNLI. Fixed Final400, no test-based selection; report negative results and paired intervals. Additional resources and calibration compute counted separately.

## Latest user extension: FFT supplement

The user supplied a new FFT task on2026-11-09, asking to supplement the current experiments. Continue the three-language LoRA formal jobs and add the independent SW FFT36 track in `research/fft_representation_20261109`. Reuse verified historic FFT, train matched unprotected FFT and FFT+original36-site representation, both full400 with non-reentrant checkpointing and a single combined backward; complete the stated formal benchmarks including both translation directions and missing HumanEval+. Do not mix original36-site regularization with the newer72-site weighted track. No new training seeds or score gate. Both new FFT jobs have been manually queued, with GPU engineering still pending.
