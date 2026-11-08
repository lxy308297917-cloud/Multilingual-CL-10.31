# P2 scientific code and reproduction boundaries

The loss extension in `losses.py` and numeric audit in `recompute_numeric.py` are
byte-identical to the executed scientific code. Other scripts adapt only repository
root discovery and external-resource/output paths; `public_source_map.json` records both hashes.
Target SFT remains LLaMA-Factory/HF Trainer; this is not a separate training framework.
`audit_replay_outputs.py` was added after the replay core result as a posthoc error
analysis. It joins saved GSM8K outputs to completion metadata by the observer's JSON
response hash, describes style/caps, and never replaces official scores.

## CPU result audit without models or raw questions

```bash
python scripts/functional_representation/p2/recompute_numeric.py \
  --root research/functional_representation_20261109/p2
```

Requires NumPy. The final numeric matrices reproduce40 absolute scores and48 paired
intervals, including the direct feature-versus-replay comparison. They do not contain
questions, responses or weights. Instruction intervals resample prompt clusters.

The existing12-case tiny-Qwen/PEFT gradient oracle can be rerun in the locked HF
environment using `python scripts/functional_representation/p2/validate_accumulation.py`.
It explicitly disables CUDA and writes `accumulation_validation.json` in the current
directory. Its check covers engineering correctness, not downstream efficacy.

## Training and input preparation

Use the locked upstream environment, Qwen/Qwen3-4B revision
`1cfa9a7208912126459214e8b04321603b3df60c`, original SW6400 mixture and labels, and
the build-only old pool with its Base feature cache. These external assets are not
packaged here. Replace `/path/to/...` in the three YAMLs explicitly; YAML values are
not automatically expanded from shell variables.

`train.py` extends the existing public `scripts/sft_v1/train_formal.py` accounting
and upstream trainer. `prepare.py` builds the P2 pool/cache from the preserved P1
encoded data and Base features; it requires those earlier resources. It does not
download or synthesize replacement data. This export is not a complete standalone
data acquisition or server orchestration distribution.

Environment paths, used only by scripts that need the corresponding resource:

| Variable | Resource |
|---|---|
| `OLD_POOL_DIR` | P2 build_pool.jsonl, build_reference_features.safetensors, manifest.json |
| `P1_DIAGNOSTIC_DATA` | Prior encoded_selected/features/Full per-example diagnostic resources |
| `P2_CHECKPOINT_ROOT` | Three independent model checkpoint directories |
| `P2_LOG_ROOT` | Endpoint outputs under evaluation/<mode>/<stage> |
| `REFERENCE_SW_LOG_ROOT` | Historical frozen Base/Full raw evaluation outputs |
| `LEGACY_FULL_ADAPTER` | Historical unprotected Full adapter and exposure/training records |
| `TARGET_EXPOSURE_REFERENCE` | Included target_exposure_reference.jsonl, or original hash ledger |
| `QWEN3_4B` | Local official model directory, for runtime protocol auditing |

Choose an actually free GPU according to the host's rules, set its UUID in
`CUDA_VISIBLE_DEVICES`, and set `AYA_EXPECTED_GPU_UUID` to the same value. Each run
is single-GPU, uses an independent nonexistent output directory, and runs400 steps.

```bash
python scripts/functional_representation/p2/train.py \
  --mode full_feature \
  --config configs/functional_representation_20261109/p2/full_feature_formal_v1.yaml
```

The modes are `all_layer`, `full_feature`, `ce_replay`. Initial adapter0 is saved;
optimizer/scheduler/RNG checkpoints are saved at50/100/200/300/400. Recovery needs a
valid optimizer checkpoint and a new output directory. GPU final tensors are not
bitwise repeatable; exact loaded-state and sample-stream checks and the numerical
repeat control are reported separately. No English-performance early stopping or
test-selected checkpoint is used.

`collect.py` requires the original raw endpoint outputs for input identity and
candidate-argmax checks. `audit_training.py` requires saved training artifacts.
`audit_protocol.py` inspects actual configs and physical prompt-token traces without
rescoring official IFEval. `costs.py` separates actual restored-process exposures,
old labels, feature positions, cache and historical normalization reference cost.
`build_results.py` is gated on complete audits; training curves require saved model
records. `update_norms.py` is a posthoc magnitude description, not a performance test
or proof of a functional mechanism.

The server-specific queue/controllers are deliberately excluded. Frozen benchmark
configurations and existing generation/scoring implementations remain in the prior
public project; the final scientific command record uses local-path placeholders.
