"""Locked standard metrics; corpus MT is never a sentence-score average."""
import collections,re,math
import numpy as np
from sacrebleu.metrics import CHRF,BLEU


def mt_metrics(predictions,references):
 if len(predictions)!=len(references) or not predictions:raise ValueError('Invalid scorer input length')
 chrf=CHRF(char_order=6,word_order=2,beta=2,lowercase=False,whitespace=False,eps_smoothing=False);bleu=BLEU(lowercase=False,tokenize='13a',smooth_method='exp',smooth_value=None,effective_order=False)
 c=chrf.corpus_score(predictions,[references]);b=bleu.corpus_score(predictions,[references])
 return {'chrF++':c.score,'BLEU':b.score},{'chrF++':str(chrf.get_signature()),'BLEU':str(bleu.get_signature())}


def rouge_rows(predictions,references):
 from rouge_score.rouge_scorer import RougeScorer
 import nltk
 scorer=RougeScorer(['rouge1','rouge2','rougeLsum'],use_stemmer=False,split_summaries=False)
 def lines(text):return '\n'.join(nltk.sent_tokenize(text.strip(),language='english'))
 # Same pinned Punkt and default rouge-score word tokenizer for both languages.
 # SW segmentation/word morphology limitations are disclosed; no cross-language absolute comparison.
 return [{'ROUGE-1':s['rouge1'].fmeasure*100,'ROUGE-2':s['rouge2'].fmeasure*100,'ROUGE-Lsum':s['rougeLsum'].fmeasure*100} for s in [scorer.score(lines(ref),lines(pred)) for pred,ref in zip(predictions,references)]]


def paired_bootstrap(predictions,base_predictions,references,metric='chrF++',repeats=1000,seed=42):
 if not (len(predictions)==len(base_predictions)==len(references)) or not references:raise ValueError('Paired lengths differ or empty')
 if repeats<1:raise ValueError('Bootstrap repeats must be positive')
 rng=np.random.default_rng(seed);n=len(references)
 def score(h,r):return mt_metrics(h,r)[0][metric]
 point=score(predictions,references)-score(base_predictions,references);deltas=[]
 for _ in range(repeats):
  ids=rng.integers(0,n,size=n);ref=[references[i] for i in ids];deltas.append(score([predictions[i] for i in ids],ref)-score([base_predictions[i] for i in ids],ref))
 low,high=np.percentile(deltas,[2.5,97.5]);return dict(delta=point,ci95_low=float(low),ci95_high=float(high),repeats=repeats,seed=seed,unit='paired original sample IDs, metric recomputed on each sampled corpus')


def repetition(text):
 words=text.split()
 for width in range(1,6):
  for start in range(max(0,len(words)-width*10+1)):
   pattern=words[start:start+width]
   if all(words[start+k*width:start+(k+1)*width]==pattern for k in range(1,10)):return True
 return False


def output_diagnostics(rows):
 from langdetect import DetectorFactory,detect_langs,LangDetectException
 DetectorFactory.seed=42;result=[]
 for row in rows:
  text=row['prediction'];target=row['direction'].split('->')[-1];letters=sum(c.isalpha() for c in text);detected=None;confidence=None;reason=None
  if target not in ('en','sw'):reason='not a natural-language task'
  elif letters<20 or len(text.split())<4:reason='too short for declared detector coverage'
  else:
   try:
    guess=detect_langs(text[:10000])[0];confidence=guess.prob
    if confidence>=.8:detected=guess.lang
    else:reason='confidence below0.8'
   except LangDetectException:reason='detector could not classify'
  result.append(dict(id=row['id'],empty=not text.strip(),exact_source_copy=text.strip()==row.get('source','').strip(),repetitive=repetition(text),generated_tokens=row['generated_tokens'],hit_cap=bool(row['generation_hit_cap']),article_truncated=bool(row.get('article_truncated',False)),input_tokens=row['input_tokens'],expected_language=target,detected_language=detected,detector_confidence=confidence,language_match=None if detected is None else detected==target,unmeasured_reason=reason,language_diagnostic_scope='langdetect1.0.9 auxiliary, not validated language ground truth or complete translation-direction correctness'))
 return result
