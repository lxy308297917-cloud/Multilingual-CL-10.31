# SST reproduction adapter

Algorithm source: Arnav0400/SST at ff0c5bfb32f621ca952d12042b046251181b8aa5, paper2610.22293v1. This is an independently adapted LLaMA-Factory/DeepSpeed3 path. No upstream trainer source is vendored. R10 follows the paper extension. Experimental parallel-SVD code failed its equivalence gate and is excluded from this formal snapshot.

Read [implementation notes](../../research/sst_reproduction/implementation_notes.md) and [frozen protocol](../../research/sst_reproduction/training_protocol.md). Configs and host-specific benchmark drivers replace private locations with `/local` placeholders; configure your own model, dataset, runtime and upstream paths. The recorded initial SVD cache and trained checkpoints are local artifacts, not included. Fresh SVD in another environment does not promise byte-identical bases.

CPU mathematical gates: `python tests/sst/test_core.py`. Bounded coefficient gates: `python -m unittest discover -s tests/research_my_method`. The actual two-rank test is `tests/sst/test_deepspeed_master.py`; it requires explicit preflight, two genuinely free GPUs and a matching environment. Inspecting these files does not start GPU jobs.

Formal training uses `python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/sst/train_sst.py --config CONFIG`, with two explicit GPU UUIDs in CUDA_VISIBLE_DEVICES and matching frozen dataset/model identities. Do not interpret the rank percentage as trainable-parameter count or compute reduction. Only load checkpoints you generated and verified.
