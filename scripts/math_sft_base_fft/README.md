# Independent multilingual mathematical SFT

This is a curated scientific snapshot of the actual local experiment, with explicit deployment profiles. Internal GPU schedulers, raw datasets/answers/predictions, weights and private host receipts are excluded. The source mapping records both actual-run and public-file hashes; path/profile changes are disclosed. Numerical training/scoring constants are checked during export.

## Entry points

| Stage | Scripts |
|---|---|
| Pinned sources and exact prompts | `fetch_sources.py`, `prepare_eval.py` |
| All valid independent training examples | `audit_mathoctopus.py`, `prepare_train.py` |
| Real two-epoch FFT and exposure proof | `make_training_configs.py`, `run_training_job.py`, `train_two_epochs.py`, `validate_training.py` |
| Frozen target mathematics | `evaluate_math.py`, `score_polymath.py` |
| Frozen English benchmarks | `evaluate_general.py`, `evaluate_checkpoint_gpu.py`, `execute_code.py`, `audit_general_scores.py` |
| Nine-language selection and paired analysis | `screen_languages.py`, `aggregate_route_a.py` |
| Actual cost, answer-free per-item results and final reports | `collect_delivery_evidence.py`, `finalize_route_a_delivery.py` |
| Bounded optional data diagnostics | `language_sample.py`, `audit_english_source_anchors.py`, `audit_output_language.py`; `prepare_engineering.py` creates the separate 33-record engineering canary |

Training uses the existing pinned LLaMA-Factory + DeepSpeed stack and accounting callbacks in `scripts/sft_v1`. No alternate Trainer or new method is introduced. Use the project's locked original MMLU task definitions, SFT GSM8K/IFEval definitions and HumanEval+ data preparation. The native dataset and prompt-token hashes are in the scientific manifests.

## Configure paths

`profile_paths.py` exposes path keys through environment variables. Set at least:

```bash
export MLCL_ROOT=/path/to/Multilingual-CL-10.31
export MATH_SFT_DATA=/path/to/math-sft-data
export MLCL_MODEL=/path/to/pinned-Qwen3-1.7B
export MLCL_TRAIN_PY=/path/to/llamafactory-env/bin/python
export MLCL_EVAL_PY=/path/to/vllm-env/bin/python
export MLCL_CODE_PY=/path/to/evalplus-env/bin/python
export MLCL_AUDIT_PY=/path/to/fasttext-numpy1-env/bin/python
export MLCL_GLOTLID=/path/to/glotlid/model_v3.bin
export MLCL_CHECKPOINT_DIR=/path/to/math-checkpoints
export MLCL_LOG_DIR=/path/to/math-logs
export MLCL_MMLU_TASKS=/path/to/original-locked-mmlu-tasks
export MLCL_CODE_DATA_ROOT=/path/to/original-bench-v2-data
export NLTK_DATA=/path/to/frozen-nltk-resources
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The source fetcher retrieves pinned public MGSM-Rev2, MSVAMP, PolyMath and its official grader. It explicitly fetches the original MathOctopus file and canonical GSM8K train dependency that were already cached on the experiment host. It never downloads another model automatically. Optional `MLCL_LEGACY_DATA` only reuses an existing cache. New rebuilds therefore have extra explicit training-source entries in their source lock; compare the actual asset/token payloads and the disclosed source mapping, rather than claiming the entire deployment-dependent JSON has identical bytes.

```bash
cd "$MLCL_ROOT"
"$MLCL_TRAIN_PY" scripts/math_sft_base_fft/fetch_sources.py
"$MLCL_TRAIN_PY" scripts/math_sft_base_fft/prepare_eval.py
"$MLCL_TRAIN_PY" scripts/math_sft_base_fft/audit_mathoctopus.py
"$MLCL_TRAIN_PY" scripts/math_sft_base_fft/prepare_train.py
```

For a new scientific run, freeze the protocol and every input token before evaluating Base. Reusing the published selected languages is reproduction of this exploratory study; it is not a fresh blind language-selection exercise.

## GPU runs

Check actual GPU processes, owners, utilization, free memory and this account's total GPUs, including other queues. Explicitly set GPU UUIDs, and do not exceed the account limit. The private run used at most four GPUs total and two GPUs per FFT training.

```bash
# Example target evaluation; math uses V1 for every current role.
CUDA_VISIBLE_DEVICES=GPU-UUID VLLM_USE_V1=1 "$MLCL_EVAL_PY" \
  scripts/math_sft_base_fft/evaluate_math.py \
  --langs sw bn ja --output /new/unique/base-output
```

Provided YAMLs retain the exact scientific values but expose path placeholders. Materialize them without shell expansion:

```bash
"$MLCL_TRAIN_PY" scripts/math_sft_base_fft/materialize_config.py \
  --input configs/math_sft_base_fft/mathoctopus_sw_full_2epochs_seed42_v1.yaml \
  --output /new/unique/runtime-config.yaml
CUDA_VISIBLE_DEVICES=GPU-UUID-A,GPU-UUID-B "$MLCL_TRAIN_PY" \
  scripts/math_sft_base_fft/run_training_job.py --config /new/unique/runtime-config.yaml
```

The runner performs **one** continuous two-epoch training invocation, then validates both epoch ledgers and independently loads each saved model for a CPU forward pass. `save_only_model=true`: weights are independently loadable, but optimizer state is not persisted. Epoch 2 is not a restarted optimizer. Do not overwrite existing outputs. Nondivisible final batches and distributed padding are measured from actual traces.

`evaluate_general.py` fixes the current English GSM8K/IFEval **and HumanEval generation** to V0. The older November 2 HumanEval Base used V1 and is not mixed into this study. Current shared Base and all six FFT nodes use the same per-benchmark engine/template/decoding. The paired outputs verify exact prompt-token identity.

## Official PolyMath and actual code execution

Set `MLCL_POLYMATH_ENV` to a separate Python3.11 environment installed from the pinned official PolyMath requirements downloaded under `sources/PolyMath-official`. The grader restores the upstream missing `symeval.EvaluatorMath` export only; math equivalence is unchanged. It uses the first balanced boxed answer, not last-number extraction. Dataset `medium` maps to official `middle`; difficulty weights are 1,2,4,8 divided by 15.

`executor_adapter/` records the trusted EvalPlus wrapper, limits, namespace launcher and the actual dispatch functions. Set `MLCL_EXECUTOR_DIR` and `MLCL_SANDBOX_WORKSPACE` to a **locally verified** deployment. Configure the read-only Python runtime, non-setuid bubblewrap0.11.0, mounts and EvalPlus0.3.1; regenerate measured host-file/network/process/memory/CPU/wall gates and the proof/runtime manifests before executing model code. These source files alone are not a ready-made isolation proof. The adapter fails when the required proof is absent. Never execute generated solutions in an ordinary shell or treat venv/reliability_guard alone as isolation. See the existing [execution report](../../research/bench_v2/code_execution.md).

The code run uses the unchanged verified 4GiB/2-process/120-second namespace policy, two parallel sandboxes, all 164 canonical references first, then real model execution. Plus pass requires both base and plus tests. The official `HumanEval/32` success bookkeeping repair preserves its original mathematical assertion.

PolyMath uses a separate derived math worker policy: 4GiB memory, 8MiB stack (so a multiprocessing queue feeder can start), four process/thread tasks, otherwise the same isolation. The trusted queue probe and six official equivalence calibrations passed. Do not apply this derived policy silently to code execution.

## Interpretation

Read [the frozen scientific protocol](../../research/math_sft_base_fft/protocol.md), [all nine Base languages](../../research/math_sft_base_fft/base_language_screening.md), and [the dataset audit](../../research/math_sft_base_fft/mathoctopus_data_audit.md). This study uses exposed benchmarks to choose languages and inspect fixed budgets. It reports single-seed paired question uncertainty, not multi-seed stability or independent blind confirmation. Neither declining English scores nor disappointing target scores cause baseline early stopping. Route B is conditional on the predeclared route-A gate and remains a separate dataset/training run.

## Reproduce the original artifact layout

The analysis scripts intentionally consume the frozen run layout. For a complete fresh census, run `evaluate_math.py` as three single-GPU jobs, using groups `sw bn th`, `ja zh ru`, and `fr de es`, with output names `results/math_sft_base_fft/base-census-worker1-v1`, `base-census-worker2-v1`, `base-census-worker3-v1`. Calibrate the configured PolyMath sandbox to `results/math_sft_base_fft/polymath-calibration-v3`, then run `score_census_once.py`; this finite CPU drain freezes each completed language and exits after all nine. Run `screen_languages.py` only after all nine score folders are complete.

Generate the shared current Base English output through `evaluate_general.py --role base --output results/math_sft_base_fft/base-general-v1`. For each actual trained epoch, `evaluate_checkpoint_gpu.py` writes `results/math_sft_base_fft/A/<language>/epoch<1-or-2>`; provide the correct full checkpoint and role name. `finish_cpu_once.py` is a finite CPU drain for the shared Base and six fixed epochs, performing official PolyMath and actual isolated EvalPlus execution. It does not start GPU jobs or select a better checkpoint. Once all roles are complete, run their independent `audit_general_scores.py` checks and `aggregate_route_a.py`. Review the actual frozen route gate before considering the separate conditional mCoT-MATH plan.

Internal host GPU admission queues and their private process/UUID state are not included in this snapshot. Schedule these explicit jobs with your own verified resource controls; the scientific scripts never authorize interfering with another user's process.

After the complete fixed matrix and independent score audits, `collect_delivery_evidence.py` exports exact compressed input metadata, training loss/LR/gradient-norm histories, measured per-rank memory peaks, observed engine/weight/protocol identity checks, and answer-free per-item flags. `audit_output_language.py` is a bounded CPU-only diagnostic with the same sampled IDs at every epoch; it never changes numeric accuracy, data filtering or route choice. Its automatic labels on mathematical text are not human-verified language correctness. Finish with `finalize_route_a_delivery.py` after both evidence groups are complete. If the actual route-A gate fails, that report explicitly leaves the conditional route-B work outstanding.
