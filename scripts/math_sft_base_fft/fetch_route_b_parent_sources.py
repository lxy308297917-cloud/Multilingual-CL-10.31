"""Fetch only two pinned English source datasets for mCoT source-group auditing."""
import concurrent.futures, json, urllib.request, time, shutil
from pathlib import Path
from common import *
SOURCES = [{'repo': 'meta-math/MetaMathQA', 'revision': 'aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18', 'file': 'MetaMathQA-395K.json', 'size': 395626321, 'sha256': 'fb39a5d8c05c042ece92eae37dfd5ea414a5979df2bf3ad3b86411bef8205725'}, {'repo': 'TIGER-Lab/MathInstruct', 'revision': 'b4fdc323a7be1379c9c7c0b67b1de72dfee2111a', 'file': 'MathInstruct.json', 'size': 212488891, 'sha256': '15b8be9824c31d619078f43baf67f7243c53f475ae4de5f8346b6d3ea9c269dd'}]

def main():
    gate = json.loads((ROOT / 'results/math_sft_base_fft/routeA-math-gate-v1/decision.json').read_text())
    assert gate['status'] == 'FAIL_FIXED_E1_AND_E2_GATE'
    out = DATA / 'mcot-source-audit-v1/source-parents'
    out.mkdir(parents=True, exist_ok=True)

    def fetch(source):
        p = out / source['file']
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['file']}?download=true"
        if p.exists():
            assert p.stat().st_size == source['size'] and sha(p) == source['sha256']
            return dict(source, status='VERIFIED_EXISTING', url=url)
        assert shutil.disk_usage(out).free > 42 * 1024 ** 3, 'Preserve full B checkpoint budget and storage margin'
        partial = p.with_suffix(p.suffix + '.partial')
        if partial.exists():
            partial.rename(partial.with_name(partial.name + '.preserved-' + str(time.time_ns())))
        digest = hashlib.sha256()
        n = 0
        start = time.monotonic()
        request = urllib.request.Request(url, headers={'User-Agent': 'Multilingual-CL-scientific-source-audit/1.0'})
        with urllib.request.urlopen(request, timeout=90) as response, partial.open('xb') as f:
            while True:
                block = response.read(1024 ** 2)
                if not block:
                    break
                f.write(block)
                digest.update(block)
                n += len(block)
        assert n == source['size'] and digest.hexdigest() == source['sha256'], (source['file'], n, digest.hexdigest())
        partial.rename(p)
        receipt = dict(source, status='COMPLETE_VERIFIED', url=url, seconds=time.monotonic() - start)
        print(json.dumps(receipt), flush=True)
        return receipt
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        receipts = list(ex.map(fetch, SOURCES))
    dump(out / 'SOURCE_MANIFEST.json', {'status': 'COMPLETE', 'sources': receipts, 'route_A_gate_sha256': sha(ROOT / 'results/math_sft_base_fft/routeA-math-gate-v1/decision.json'), 'not_training_data_by_itself': 'Original English records are used for provenance/grouping, not English SFT.'})
if __name__ == '__main__':
    main()
