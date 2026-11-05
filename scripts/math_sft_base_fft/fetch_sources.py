from profile_paths import resolve
import json, hashlib, urllib.request, concurrent.futures, shutil, time
from pathlib import Path
D = Path(resolve('DATA'))
S = D / 'sources'
S.mkdir(parents=True, exist_ok=True)
OLD = Path(resolve('LEGACY_DATA') + '/sources')
langs = {'sw': 'Swahili', 'bn': 'Bengali', 'th': 'Thai', 'ja': 'Japanese', 'zh': 'Chinese', 'ru': 'Russian', 'fr': 'French', 'de': 'German', 'es': 'Spanish'}

def get(url):
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Multilingual-CL research'}), timeout=90) as r:
                return r.read()
        except Exception:
            if k == 2:
                raise
            time.sleep(2)

def asset(job):
    rel, url, old = job
    p = S / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        if old and old.exists():
            shutil.copyfile(old, p)
        else:
            b = get(url)
            t = p.with_suffix(p.suffix + '.partial')
            t.write_bytes(b)
            t.replace(p)
    b = p.read_bytes()
    return {'path': str(rel), 'url': url, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
meta = json.loads(get('https://huggingface.co/api/datasets/Qwen/PolyMath'))
rev = meta['sha']
(S / 'polymath_metadata.json').write_text(json.dumps(meta))
git = json.loads(get('https://api.github.com/repos/QwenLM/PolyMath/commits/main'))
g = git['sha']
tree = json.loads(get('https://api.github.com/repos/QwenLM/PolyMath/git/trees/' + g + '?recursive=1'))
(S / 'polymath_git_tree.json').write_text(json.dumps(tree))
ms = '301e2b3b168be70058c89c21c2fbdc9262102add'
mg = '5f091edeae66ebc605ac4f98dca07a6c345f0060'
jobs = []
for l, n in langs.items():
    rel = Path('mgsm_rev2') / ('mgsm_' + l + '.tsv')
    jobs.append((rel, f'https://raw.githubusercontent.com/google-research-datasets/MGSM-Rev2/{mg}/MGSM-Rev2/mgsm_{l}.tsv', OLD / 'google-research-datasets--MGSM-Rev2' / rel.name))
    rel = Path('msvamp') / ('test_' + n + '.json')
    jobs.append((rel, f'https://huggingface.co/datasets/Mathoctopus/MSVAMP/resolve/{ms}/{rel.name}', OLD / 'Mathoctopus--MSVAMP' / rel.name))
    for level in ['low', 'medium', 'high', 'top']:
        rel = Path('polymath') / l / (level + '.parquet')
        jobs.append((rel, f'https://huggingface.co/datasets/Qwen/PolyMath/resolve/{rev}/{l}/{level}.parquet', None))
jobs.append((Path('mathoctopus') / 'MGSM8KInstruct_Parallel.json', 'https://huggingface.co/datasets/Mathoctopus/GSM8KInstruct_Parallel/resolve/dd8076b83bacd7c6ade2829317e8a500c6ffbec4/MGSM8KInstruct_Parallel.json', Path(resolve('MATH_SOURCE'))))
jobs.append((Path('gsm8k') / 'train-00000-of-00001.parquet', 'https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet', Path(resolve('GSM8K_TRAIN'))))
for x in tree['tree']:
    if x['type'] == 'blob' and (not x['path'].startswith('ASSETS/')):
        rel = Path('PolyMath-official') / x['path']
        jobs.append((rel, f'https://raw.githubusercontent.com/QwenLM/PolyMath/{g}/' + x['path'], None))
records = []
fail = []
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    fs = {ex.submit(asset, j): j for j in jobs}
    for f in concurrent.futures.as_completed(fs):
        try:
            r = f.result()
            records.append(r)
            print('OK', r['path'], r['bytes'], flush=True)
        except Exception as e:
            fail.append({'path': str(fs[f][0]), 'error': repr(e)})
            print('FAILED', fail[-1], flush=True)
lock = {'polymath_dataset_revision': rev, 'polymath_code_revision': g, 'mgsm_revision': mg, 'msvamp_revision': ms, 'assets': sorted(records, key=lambda x: x['path']), 'failures': fail}
(D / 'sources.lock.json').write_text(json.dumps(lock, indent=2) + '\n')
assert not fail, fail
