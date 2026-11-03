# Bounded multilingual mechanism study

The implementation is a finite perturbation study, **not a completed new continual-learning training method**. `coefficient_geometry.py` and `pilot_semantic_coefficients.py` provide an orthonormal acquired-update coordinate branch; all original weights stay frozen. A complete compatibility controller and the B/C draft algorithms are not implemented.

Order: audited language source → `prepare_hausa_sft.py` / `prepare_sequential_dev.py` → freeze protocol with `prepare_coefficient_protocol.py` → `build_update_mode_bank.py` from a verified native400-step SST checkpoint → engineering fit-only check → one full `pilot_semantic_coefficients.py` run → `analyze_coefficient_pilot.py`. The main contrast, fit/confirmation IDs, strengths and controls were frozen before outputs. Raw questions/token IDs, gradient tensors and mode banks are not in this curated repository; reconstruct from locked upstream sources and your own verified checkpoints.

`evaluate_sequential_dev.py` is a separate exploratory0-shot FP32 evaluation. `evaluate_sequential_fft.py` records fixed full-test reporting; it never selects a language/model. `probe_tied_embedding.py` is an additional preregistered finite restore/random control. Host-specific formal drivers use explicit `/local` placeholders requiring configuration.

`audit_*` and plotting/collection scripts are read-only analyses of completed artifacts, except for writing independent result directories. Results and interpretation: [research decision](../../research/my_method_20261103/final_recommendation.md). This snapshot introduces no new full-method training.
