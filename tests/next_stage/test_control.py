import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/next_stage'))
import numpy as np
from analyze_ha_control import point_contrasts
from analyze_curves import interval
class Control(unittest.TestCase):
 def test_gain_difference_is_not_final_gap(self):
  v={'A0':[0,0,1,0],'A1':[0,1,1,0],'B0':[1,0,1,0],'B1':[0,1,1,0]};d=point_contrasts(v);self.assertEqual(float(d['final_A_minus_B'].mean()),0.);self.assertEqual(float(d['A_gain_minus_B_gain'].mean()),.25);self.assertTrue(np.array_equal(d['A_gain_minus_B_gain'],d['final_A_minus_B']+d['initial_B_minus_A']))
 def test_cluster_weighting_uses_micro_denominator(self):
  d=interval([1,1,0],[1,1,1],['passage1','passage1','passage2'],100);self.assertEqual(d['clusters'],2);self.assertAlmostEqual(d['delta'],2/3);self.assertTrue(d['ci95_low']<=d['delta']<=d['ci95_high'])
if __name__=='__main__':unittest.main()
