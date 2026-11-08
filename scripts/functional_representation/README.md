# Functional representation diagnostic code

`representation.py` implements frozen block-output capture, assistant-position CE,
feature gradients and rank-matched bases. It is a diagnostic, not a training method.
The original CPU toy-Qwen validation is retained in the research report.

`analyze_diagnosis.py` and `analyze_covariates.py` reproduce the exploratory build/check
statistics with the frozen plans in `configs/functional_representation_20261109`.
Set `FUNCTIONAL_DIAGNOSTIC_DATA` to the local encoded/feature-cache directory and
`FUNCTIONAL_BEHAVIOR_LOG` to the local calibration prediction directory. Run the
primary analysis before the covariate supplement. Their output destination is
`research/functional_representation_20261109`.

Raw calibration questions, generated answers, feature caches and weights are not
distributed here. The public numeric tables and hashes permit result inspection;
this export is not a complete standalone GPU orchestration package. No model training
has run in this phase. Source input identities, split checks and numerical checks
are reported separately. Original scoring labels are preserved despite the posthoc
answer-format limitation.
