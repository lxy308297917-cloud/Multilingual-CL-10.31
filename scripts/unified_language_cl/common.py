"""Shared data identity and task prompts for the unified language experiment."""
from runtime import runtime_path
import hashlib, json, re, unicodedata
from pathlib import Path
from decimal import Decimal, InvalidOperation
ROOT = Path(runtime_path('project', ''))
DATA = Path(runtime_path('data', ''))
MODEL = Path(runtime_path('model', ''))
LANGS = ['es', 'th', 'fr', 'en']
NAMES = {'es': 'Spanish', 'th': 'Thai', 'fr': 'French', 'en': 'English'}

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

def write_rows(p, rs):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    assert not p.exists(), p
    p.write_text(''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in rs)))

def norm(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())

def order(ids, salt):
    return sorted(ids, key=lambda i: oh([42, salt, str(i)]))

def number(s, lang='en'):
    """Canonical decimal, conservative final-number extraction, never run code."""
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

def numeric_prediction(text):
    strict = re.findall('####\\s*([-+]?\\d[\\d,]*(?:\\.\\d+)?)', text)
    return number(strict[-1]) if strict else None

def prompt(track, lang, text, labels=None):
    if track == 'massive':
        ins = {'en': 'Classify the intent of the utterance. Output only one label from this list: ', 'es': 'Clasifica la intención del enunciado. Devuelve únicamente una etiqueta de esta lista: ', 'fr': 'Classez l’intention de l’énoncé. Répondez uniquement par une étiquette de cette liste : ', 'th': 'จำแนกเจตนาของข้อความ ตอบด้วยป้ายกำกับเพียงหนึ่งรายการจากรายการนี้: '}[lang] + ', '.join(labels)
    elif track == 'xnli':
        ins = {'en': 'Given the premise and hypothesis, output only entailment, neutral, or contradiction.', 'es': 'Dadas la premisa y la hipótesis, devuelve únicamente entailment, neutral o contradiction.', 'fr': 'Étant donné la prémisse et l’hypothèse, répondez uniquement par entailment, neutral ou contradiction.', 'th': 'จากข้อความตั้งต้นและสมมติฐาน ให้ตอบเพียง entailment, neutral หรือ contradiction เท่านั้น'}[lang]
    elif track == 'sib200':
        ins = {'en': 'Classify the topic. Output only one label from this list: ', 'es': 'Clasifica el tema. Devuelve únicamente una etiqueta de esta lista: ', 'fr': 'Classez le sujet. Répondez uniquement par une étiquette de cette liste : ', 'th': 'จำแนกหัวข้อ ตอบด้วยป้ายกำกับเพียงหนึ่งรายการจากรายการนี้: '}[lang] + ', '.join(labels)
    elif track == 'mathoctopus':
        ins = {'en': 'Solve the math problem in English. Explain your reasoning and finish with "#### <number>".', 'es': 'Resuelve el problema de matemáticas en español. Explica tu razonamiento y termina con "#### <número>".', 'fr': 'Résolvez le problème de mathématiques en français. Expliquez votre raisonnement et terminez par "#### <nombre>".', 'th': 'แก้โจทย์คณิตศาสตร์เป็นภาษาไทย อธิบายวิธีคิดและลงท้ายด้วย "#### <ตัวเลข>"'}[lang]
    elif track == 'mtop':
        ins = {'en': 'Parse the utterance into a compositional decoupled semantic tree. Output only the tree using IN: intent and SL: slot labels. Schema: ', 'es': 'Analiza el enunciado en un árbol semántico composicional desacoplado. Devuelve solo el árbol con etiquetas IN: y SL:. Esquema: ', 'fr': 'Analysez l’énoncé en un arbre sémantique compositionnel découplé. Répondez uniquement par l’arbre avec les étiquettes IN: et SL:. Schéma : ', 'th': 'แปลงข้อความเป็นต้นไม้การวิเคราะห์ความหมายแบบ compositional decoupled ตอบเฉพาะต้นไม้โดยใช้ป้าย IN: และ SL: สคีมา: '}[lang] + ', '.join(labels)
        ins += '\nSyntax: [IN:INTENT_NAME [SL:SLOT_NAME slot value ] ]. Use square brackets and actual schema labels; keep slot values in the utterance language. No XML, code fences, explanations, or extra text.'
    else:
        ins = {'en': 'Summarize the following article in English. Output only the summary.', 'es': 'Resume el siguiente artículo en español. Devuelve únicamente el resumen.', 'fr': 'Résumez l’article suivant en français. Répondez uniquement par le résumé.', 'th': 'สรุปบทความต่อไปนี้เป็นภาษาไทย ตอบเฉพาะบทสรุปเท่านั้น'}[lang]
    return ins + '\n\n' + text

def messages(r):
    return [{'role': 'user', 'content': r['prompt']}]
