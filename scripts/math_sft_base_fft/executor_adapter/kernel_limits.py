"""Hard kernel limits and libseccomp policy for a non-root namespace worker."""
import ctypes, errno, os, resource
MEMORY = 4 * 1024 ** 3
DENY_COMMON = ['unshare', 'setns', 'mount', 'mount_setattr', 'umount2', 'pivot_root', 'chroot', 'ptrace', 'process_vm_readv', 'process_vm_writev', 'pidfd_getfd', 'bpf', 'perf_event_open', 'userfaultfd', 'io_uring_setup', 'keyctl', 'add_key', 'request_key', 'open_by_handle_at', 'name_to_handle_at', 'init_module', 'finit_module', 'delete_module', 'kexec_load', 'kexec_file_load', 'reboot', 'swapon', 'swapoff', 'sched_setaffinity', 'pidfd_send_signal', 'rt_sigqueueinfo', 'rt_tgsigqueueinfo']

def seccomp(deny):
    lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(2147418112)
    if not ctx:
        raise RuntimeError('seccomp_init failed')
    try:
        for name in deny:
            nr = lib.seccomp_syscall_resolve_name(name.encode())
            if nr >= 0 and lib.seccomp_rule_add(ctx, 327680 | errno.EPERM, nr, 0) != 0:
                raise RuntimeError('seccomp rule failed: ' + name)
        if lib.seccomp_load(ctx) != 0:
            raise RuntimeError('seccomp_load failed')
    finally:
        lib.seccomp_release(ctx)

def driver_limits(cpu_seconds=110):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise RuntimeError('disable dumpability failed')
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise RuntimeError('no_new_privs failed')
    for name, value in [('RLIMIT_AS', MEMORY), ('RLIMIT_DATA', MEMORY), ('RLIMIT_STACK', MEMORY), ('RLIMIT_NPROC', 2), ('RLIMIT_CPU', cpu_seconds), ('RLIMIT_FSIZE', 8 * 1024 ** 2), ('RLIMIT_NOFILE', 64), ('RLIMIT_CORE', 0)]:
        resource.setrlimit(getattr(resource, name), (value, value))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    seccomp(DENY_COMMON)

def child_limits():
    seccomp(DENY_COMMON + ['clone', 'clone3', 'fork', 'vfork', 'execve', 'execveat', 'kill', 'tkill', 'tgkill', 'socket', 'socketpair', 'connect', 'bind', 'listen', 'accept', 'accept4', 'prlimit64', 'setrlimit'])
