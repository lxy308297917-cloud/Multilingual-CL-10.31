"""Fixed per-role bounds for the isolated EvalPlus CPU executor only.

This does not change the project's default container policy or add a general
memory-budget override. Official EvalPlus memory is4GiB per process. PID1 is
Python itself; at most one evaluator child exists. All processes inherit hard
address-space limits, and writable tmpfs totals80MiB. Two concurrent jobs use
less than17GiB in this500GiB machine, with no GPU allocation.
"""
import re
PROCESS_MEMORY = 4 * 1024 ** 3
PROCESSES = 2
TMPFS_MEMORY = 80 * 1024 ** 2
JOB_MEMORY = PROCESSES * PROCESS_MEMORY + TMPFS_MEMORY

def require_fixed_namespace_proof(proof):
    if proof.get('executor') != 'equivalent_verified_sandbox':
        raise ValueError('Wrong executor')
    for key in ['host_files_inaccessible', 'network_block_test_passed', 'resource_enforcement_test_passed']:
        if proof.get(key) is not True:
            raise ValueError('Missing measured gate: ' + key)
    p = proof['policy']
    required = {'network': 'none', 'read_only_root': True, 'privileged': False, 'host_pid': False, 'host_ipc': False, 'no_new_privileges': True, 'cap_drop_all': True, 'devices': [], 'sensitive_host_mounts': [], 'limits_measured_in_kernel': True, 'memory_bytes': JOB_MEMORY, 'pids_limit': PROCESSES, 'per_process_memory_bytes': PROCESS_MEMORY, 'writable_tmpfs_bytes': TMPFS_MEMORY, 'cpus': 2, 'per_task_wall_seconds': 120, 'as_pid_1': True}
    for key, value in required.items():
        if p.get(key) != value:
            raise ValueError('Fixed namespace policy mismatch: ' + key)
    if not isinstance(p.get('uid'), int) or p['uid'] <= 0:
        raise ValueError('Non-root UID required')
    if re.fullmatch('sha256:[0-9a-f]{64}', p.get('image_digest', '')) is None:
        raise ValueError('Actual read-only runtime manifest digest required')
    if p.get('image_digest_kind') != 'actual_readonly_filesystem_manifest_sha256_not_OCI':
        raise ValueError('Runtime digest kind must be explicit')
    return True
