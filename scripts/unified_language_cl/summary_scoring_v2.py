"""Pinned multilingual ROUGE with explicit tokenizer routing for sentence LCS.

Upstream afcd803's rougeLsum branch omits its configured tokenizer argument.
ROUGE-1/2 are unchanged. Preserve upstream LCS/sentence/precision algorithms;
only pass the same pinned language tokenizer to each summary sentence.
"""
import os,sys
from pathlib import Path
SCORER=Path(os.environ['XLSUM_ROUGE_ROOT'])
ADDONS=Path(os.environ['METRICS_ADDONS_ROOT'])
NLTK=Path(os.environ['NLTK_DATA'])
sys.path.insert(0,str(ADDONS));sys.path.insert(0,str(SCORER))
import nltk
nltk.data.path=[str(NLTK)]
from rouge_score.rouge_scorer import RougeScorer,_summary_level_lcs,_create_ngrams,_score_ngrams
from rouge_score import tokenization_wrapper
import rouge_score
assert Path(rouge_score.__file__).resolve().is_relative_to(SCORER.resolve())
NAMES={'es':'spanish','th':'thai','fr':'french','en':'english'}
def make_scorer(lang):return RougeScorer(['rouge1','rouge2'],use_stemmer=lang!='th',lang=NAMES[lang])
def lsum_score(scorer,reference,prediction,reference_tokens=None,prediction_tokens=None):
    def sentences(text,tokens):
        if '\n' not in text and tokens is not None:return [tokens] if len(text) else []
        return [tokenization_wrapper.tokenize(line,scorer._stemmer,scorer._tokenizer) for line in text.split('\n') if len(line)]
    return _summary_level_lcs(sentences(reference,reference_tokens),sentences(prediction,prediction_tokens))
def score_rows(preds,refs,lang):
    scorer=make_scorer(lang);result=[]
    for pred,ref in zip(preds,refs):
        ref_tokens=tokenization_wrapper.tokenize(ref,scorer._stemmer,scorer._tokenizer);pred_tokens=tokenization_wrapper.tokenize(pred,scorer._stemmer,scorer._tokenizer)
        scores={'rouge'+str(n):_score_ngrams(_create_ngrams(ref_tokens,n),_create_ngrams(pred_tokens,n)) for n in [1,2]}
        scores['rougeLsum']=lsum_score(scorer,ref,pred,ref_tokens,pred_tokens);result.append({k:100*v.fmeasure for k,v in scores.items()})
    return result
def self_test():
    examples={'en':('alpha beta gamma','alpha beta delta'),'es':('alpha beta gamma','alpha beta delta'),'fr':('alpha beta gamma','alpha beta delta'),'th':('ฉันรักแมว','ฉันรักหมา')};checks=0
    for lang,(ref,pred) in examples.items():
        same=score_rows([ref],[ref],lang)[0];assert all(abs(v-100)<1e-9 for v in same.values());checks+=1
        empty=score_rows([''],[ref],lang)[0];assert all(v==0 for v in empty.values());checks+=1
        partial=score_rows([pred],[ref],lang)[0];assert abs(partial['rougeLsum']-100*2/3)<1e-9,(lang,partial);assert abs(partial['rouge1']-100*2/3)<1e-9;assert abs(partial['rouge2']-50)<1e-9;checks+=1
        a='alpha beta\ngamma delta';b='gamma delta\nalpha beta';reordered=score_rows([b],[a],lang)[0];assert abs(reordered['rougeLsum']-100)<1e-9,(lang,reordered);checks+=1
    print('Corrected multilingual ROUGE:',checks,'identity/empty/partial-overlap/sentence-order oracles passed; Thai partial Lsum=66.6667, old upstream=0.')
if __name__=='__main__':self_test()
