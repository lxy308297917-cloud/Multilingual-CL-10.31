import unittest
from analyze_interventions import equivalence_clusters,clustered_effects,interaction_rows
class EquivalenceTests(unittest.TestCase):
 def row(self,id,correct,pair,hash):return {'id':id,'pair_id':pair,'task':'reading','input_language':'en','output_language':'en','prompt_token_sha256':hash,'max_new_tokens':2,'score':{'content_correct':correct,'format_passed':None,'language_correct':None}}
 def test_identical_prompts_collapsed_and_point_preserved(self):
  f=[self.row('a',True,'p','same'),self.row('b',True,'q','same'),self.row('c',False,'r','other')];x=[dict(r,score=dict(r['score'],content_correct=False)) for r in f];c=equivalence_clusters(f);self.assertEqual(c['a'],c['b']);self.assertNotEqual(c['a'],c['c']);r=clustered_effects(f,x,100)[0];self.assertEqual(r['clusters'],2);self.assertAlmostEqual(r['effect_final_minus_intervened_pp'],200/3)
 def test_interaction_reports_norm_confound(self):
  f=[self.row('a',True,'p','one')];a=[self.row('a',True,'p','one')];m=[self.row('a',True,'p','one')];b=[self.row('a',False,'p','one')];r=interaction_rows(f,a,m,b,True,100)[0];self.assertEqual(r['contrast_pp'],100);self.assertIn('not a pure interaction',r['interpretation'])
