"""Check one GPU and this account's GPU count before an authorized job.

The check is advisory: another user can acquire a GPU after it returns.
Never uses or terminates GPU processes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def query(fields: str, kind: str) -> list[list[str]]:
    result = subprocess.run(
        ['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True,
    )
    return [[part.strip() for part in line.split(',')] for line in result.stdout.splitlines() if line.strip()]


def process_uid(pid: int) -> int:
    return Path(f'/proc/{pid}').stat().st_uid


def snapshot(args) -> dict:
    cards = {
        row[1]: {'index': int(row[0]), 'uuid': row[1], 'free_mib': int(row[2]), 'used_mib': int(row[3]), 'util_percent': int(row[4])}
        for row in query('index,uuid,memory.free,memory.used,utilization.gpu', 'gpu')
    }
    if args.uuid not in cards:
        raise ValueError(f'Unknown GPU UUID: {args.uuid}')
    jobs = []
    for row in query('pid,gpu_uuid,used_gpu_memory,process_name', 'compute-apps'):
        pid = int(row[0])
        try:
            uid = process_uid(pid)
        except FileNotFoundError:
            raise ValueError(f'Cannot verify process owner for GPU PID {pid}; retry preflight')
        jobs.append({'pid': pid, 'uuid': row[1], 'uid': uid, 'memory_mib': int(row[2]), 'name': row[3]})
    own = sorted({job['uuid'] for job in jobs if job['uid'] == os.getuid()})
    target = cards[args.uuid]
    errors = []
    if any(job['uuid'] == args.uuid for job in jobs):
        errors.append('target GPU already has compute processes')
    if target['free_mib'] < args.min_free_mib:
        errors.append('insufficient free memory')
    if target['util_percent'] > 5:
        errors.append('target GPU utilization exceeds 5%')
    if len(set(own) | {args.uuid}) > args.max_user_gpus:
        errors.append('account GPU count would exceed configured limit')
    return {'target': target, 'own_gpu_uuids': own, 'all_jobs': jobs,
            'max_user_gpus': args.max_user_gpus, 'errors': errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--uuid', required=True)
    parser.add_argument('--min-free-mib', type=int, default=20000)
    parser.add_argument('--max-user-gpus', type=int, default=4)
    parser.add_argument('--settle-seconds', type=float, default=0,
                        help='Wait only for residual utilization when no process/memory/count conflict exists')
    args = parser.parse_args()
    if not 0 <= args.settle_seconds <= 60:
        parser.error('--settle-seconds must be in 0..60')
    deadline = time.monotonic() + args.settle_seconds
    utilization_samples = []
    while True:
        result = snapshot(args)
        utilization_samples.append(result['target']['util_percent'])
        # Fail immediately for another compute process, low memory or account limit.
        if result['errors'] != ['target GPU utilization exceeds 5%'] or time.monotonic() >= deadline:
            break
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    if args.settle_seconds:
        result['settle_utilization_samples'] = utilization_samples
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(bool(result['errors']))


if __name__ == '__main__':
    sys.exit(main())
