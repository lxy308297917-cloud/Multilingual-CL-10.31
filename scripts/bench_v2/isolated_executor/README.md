# Isolated EvalPlus execution snapshot

This records the trusted scorer, exact hard resource/seccomp helpers, fixed namespace policy, and namespace launcher used for the complete four-model execution. EvalPlus0.3.1 is pinned. `evaluate_one.py` applies one strictly matched find_zero success-accounting repair and retains the original oracle assertion.

The launcher is a host-specific reference with private environment paths replaced by placeholders. It is **not an automatically portable execution entrypoint**: set the paths, build/verify non-setuid bubblewrap0.11.0, construct a minimal read-only Python runtime, inspect every mount, and rerun the measured host-file/network/memory/process/CPU/wall gates plus correct/wrong/syntax/loop calibrations before model code. Do not claim that venv or reliability_guard alone provides isolation. Local source hashes are recorded in source-lock.json. No generated program is run when importing or inspecting these reference files.

The recorded full run used a namespace user with no capabilities, Python as PID1, no network/devices/host mounts, two CPU affinity, 4GiB hard per-process memory, at most two processes, 80MiB bounded writable tmpfs, and an outer120s limit. Hard limits and syscall behavior were measured; cgroup enforcement is not claimed. The locked runtime is a read-only file-manifest SHA256, not an OCI image digest. Every official canonical answer passed the same policy before scoring model outputs.

Scientific protocol, limitations, scores, paired task errors and verification counts: [code execution report](../../../research/bench_v2/code_execution.md).
