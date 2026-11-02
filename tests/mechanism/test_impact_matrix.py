import unittest
from build_impact_matrix import cluster_interval,impact_rows
class MatrixTests(unittest.TestCase):
 def test_paired_sign_and_clusters(self):
  point,low,high,n=cluster_interval([1,1,-1,-1],['a','a','b','b'],1000);self.assertEqual(point,0);self.assertEqual(n,2);self.assertEqual(low,-1);self.assertEqual(high,1)
 def test_protocol_and_denominators(self):
  def row(id,correct):return {'id':id,'pair_id':id,'task':'math','input_language':'en','output_language':'en','prompt_token_sha256':'fixed','max_new_tokens':2,'score':{'content_correct':correct,'format_passed':None,'language_correct':None}}
  result=impact_rows([row('1',False),row('2',False)],[row('1',True),row('2',True)],100)[0];self.assertEqual(result['effect_final_minus_intervened_pp'],-100)
  bad=row('1',True);bad['prompt_token_sha256']='changed'
  with self.assertRaises(ValueError):impact_rows([row('1',False)],[bad])
