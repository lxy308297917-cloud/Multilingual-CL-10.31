import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/bench_v2'))
import unittest
from diagnose import rank,spearman
class RankTest(unittest.TestCase):
 def test_ties(self):self.assertEqual(rank([5,1,1,3]).tolist(),[3,0.5,0.5,2])
 def test_correlation(self):
  self.assertAlmostEqual(spearman([1,2,3],[5,3,1]),-1)
  self.assertIsNone(spearman([1,1,1],[1,2,3]))
if __name__=='__main__':unittest.main()
