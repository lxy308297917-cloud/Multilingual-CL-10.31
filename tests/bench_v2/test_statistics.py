import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/bench_v2'))
import unittest
import numpy as np
from paired_statistics import paired_corpus, paired_macro
from sacrebleu.metrics import CHRF, BLEU

class StatisticsTest(unittest.TestCase):
 def test_fast_bootstrap_matches_actual_resampled_corpora(self):
  refs=['the cat sat on the mat', 'a very different second sentence', 'hello world', 'one two three four five']
  h=['the cat is on a mat','a second sentence','hello there','one two three five']
  b=['a cat on mat','the second long sentence','hi world','one two four five']
  for name,scorer in [('chrF++',CHRF(word_order=2)),('BLEU',BLEU(tokenize='13a',effective_order=False))]:
   actual=paired_corpus(h,b,refs,name,repeats=31,seed=17);rng=np.random.default_rng(17);ds=[]
   for _ in range(31):
    ids=rng.integers(0,len(refs),size=len(refs));r=[refs[i] for i in ids]
    ds.append(scorer.corpus_score([h[i] for i in ids],[r]).score-scorer.corpus_score([b[i] for i in ids],[r]).score)
   low,high=np.percentile(ds,[2.5,97.5]);self.assertAlmostEqual(actual['ci95_low'],low,places=12);self.assertAlmostEqual(actual['ci95_high'],high,places=12)
 def test_identity(self):
  self.assertEqual(paired_corpus(['a b'],['a b'],['a b'],repeats=10)['ci95_high'],0)
  self.assertEqual(paired_macro([1,2],[1,2],repeats=10)['delta'],0)
 def test_invalid(self):
  for fn,args in [(paired_corpus,([],[],[])),(paired_macro,([1],[1,2])),(paired_macro,([float('nan')],[0]))]:
   with self.assertRaises(ValueError):fn(*args)

if __name__=='__main__':unittest.main()
