"""Frozen independent nine-language mathematical SFT experiment identities."""
from profile_paths import resolve
import hashlib, json, re, unicodedata, os
from pathlib import Path
from decimal import Decimal, InvalidOperation
ROOT = Path(os.environ.get('MLCL_ROOT', resolve('ROOT')))
DATA = Path(os.environ.get('MATH_SFT_DATA', resolve('DATA')))
MODEL = Path(resolve('MODEL'))
LANGS = {'sw': 'Swahili', 'bn': 'Bengali', 'th': 'Thai', 'ja': 'Japanese', 'zh': 'Chinese', 'ru': 'Russian', 'fr': 'French', 'de': 'German', 'es': 'Spanish'}
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def oh(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def dump(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')
    t.replace(p)

def rows(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]

def norm(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())

def number(s, lang='en'):
    s = unicodedata.normalize('NFKC', str(s)).replace('−', '-').replace('\u202f', '').replace('\xa0', '')
    xs = re.findall('[-+]?\\d[\\d,]*(?:\\.\\d+)?', s)
    if not xs:
        return None
    t = xs[-1].rstrip(',')
    if ',' in t:
        if re.fullmatch('[-+]?\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?', t):
            t = t.replace(',', '')
        elif lang in ['es', 'fr'] and t.count(',') == 1 and ('.' not in t):
            t = t.replace(',', '.')
        else:
            return None
    try:
        return str(Decimal(t).normalize())
    except InvalidOperation:
        return None

def strict_number(s):
    xs = re.findall('####\\s*([-+]?\\d[\\d,]*(?:\\.\\d+)?)', s)
    return number(xs[-1]) if xs else None

def math_prompt(lang, text):
    return f'Solve the following math problem in {LANGS[lang]}. Show your reasoning. At the end, write #### followed by only the final numeric answer.\n\n' + text
