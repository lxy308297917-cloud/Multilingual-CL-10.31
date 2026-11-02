"""Independent diagnostic probe schema and transparent rule scoring.

These scores are not official benchmark results. Missing content/language
labels remain null rather than being inferred from formatting success.
"""
import json,re,unicodedata
from decimal import Decimal,InvalidOperation
from pathlib import Path


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def load_probes(path):
    with Path(path).open() as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    seen=set()
    for r in rows:
        required={'id','pair_id','task','input_language','output_language','prompt','answer','scoring','source','quality'}
        if not required.issubset(r):raise ValueError('Missing schema fields')
        if r['id'] in seen:raise ValueError('Duplicate probe id')
        seen.add(r['id'])
        if r['input_language'] not in ('en','sw') or r['output_language'] not in ('en','sw'):raise ValueError('Unsupported language')
        if not isinstance(r['prompt'],str) or not r['prompt'].strip():raise ValueError('Empty prompt')
        source=r['source']
        if not {'dataset','version','split','original_id','role'}.issubset(source):raise ValueError('Incomplete provenance')
        if source['role'] not in ('independent_development','synthetic_smoke') or source['split'] in ('test','devtest','validation_test'):raise ValueError('Formal test not permitted for module selection')
        if not isinstance(r['quality'],dict) or 'status' not in r['quality'] or 'parallel_verified' not in r['quality']:raise ValueError('Missing data quality state')
        if r['scoring']['kind'] not in ('choice','aliases','number','constraints'):raise ValueError('Unknown scoring rule')
        if r['scoring']['kind']=='choice':
            if r['answer'] not in ('A','B','C','D') or len(r.get('choices',[]))!=4:raise ValueError('Invalid choice label')
        if r['scoring']['kind']=='constraints' and not r['scoring'].get('constraints'):raise ValueError('No instruction constraints')
    if not rows:raise ValueError('Empty probe data')
    return rows


def constraint_results(text,rules):
    results=[]
    for r in rules:
        k=r['kind'];v=r['value']
        if k=='exact_words':ok=len(text.split())==v
        elif k=='contains':ok=normalized(v) in normalized(text)
        elif k=='not_contains':ok=normalized(v) not in normalized(text)
        elif k=='fullmatch':ok=re.fullmatch(v,text.strip()) is not None
        elif k=='endswith':ok=text.rstrip().endswith(v)
        else:raise ValueError(f'Unknown constraint {k}')
        results.append({'rule':r,'passed':bool(ok)})
    return results


def score_response(row,text,choice_scores=None):
    spec=row['scoring'];kind=spec['kind'];result={'content_correct':None,'format_passed':None,'constraints':[],'language_correct':None,'language_measurement':'not measured by generic detector'}
    if kind=='choice':
        if choice_scores is None or len(choice_scores)!=4:raise ValueError('Require four likelihood scores, not parsed output')
        import math
        if any(not math.isfinite(float(v)) for v in choice_scores):raise ValueError('Nonfinite option score')
        pred=max(range(4),key=lambda i:choice_scores[i]);result.update(predicted_option='ABCD'[pred],option_scores=list(choice_scores),content_correct=('ABCD'[pred]==row['answer']))
    elif kind=='aliases':result['content_correct']=normalized(text) in {normalized(a) for a in row['answer']}
    elif kind=='number':
        pattern=spec.get('strict_pattern',r'####\s*(-?\d+(?:\.\d+)?)\s*$');strict=re.search(pattern,text);result['format_passed']=strict is not None
        # Independently reported explicit answer markers; arbitrary last-number guessing is disallowed.
        candidates=re.findall(r'(?:####|<answer>)\s*(-?\d+(?:\.\d+)?)\s*(?:</answer>)?',text)
        candidate=candidates[-1] if candidates else (text.strip() if re.fullmatch(r'-?\d+(?:\.\d+)?',text.strip()) else None)
        try:result['content_correct']=candidate is not None and Decimal(candidate)==Decimal(str(row['answer']))
        except InvalidOperation:result['content_correct']=False
    elif kind=='constraints':
        result['constraints']=constraint_results(text,spec['constraints']);result['format_passed']=all(r['passed'] for r in result['constraints'])
        if row['answer'] is not None:result['content_correct']=normalized(text) in {normalized(a) for a in row['answer']}
    if 'language_reference_tokens' in spec:
        reference=spec['language_reference_tokens'];result['language_correct']=all(normalized(t) in normalized(text) for t in reference);result['language_measurement']='explicit reference-token diagnostic; not general language identification'
    return result
