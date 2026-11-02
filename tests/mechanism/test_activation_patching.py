import unittest,torch
from test_weight_delta import model
from activation_patching import prompt_activations,patched_ll,norm_matched_noise
from inference import continuation_loglikelihood
class FunctionalPatchTests(unittest.TestCase):
 def test_teacher_forced_identity_and_norm_control(self):
  m=model();names=['model.layers.1.self_attn','model.layers.1.mlp'];prefix=[1,2,3];answer=[4,5]
  data=prompt_activations(m,prefix,answer,names);before=continuation_loglikelihood(m,prefix,answer)[0]
  for name in names:
   after=patched_ll(m,prefix,answer,name,data[name]);self.assertAlmostEqual(before,after,places=6)
   donor=data[name]+.01;control=norm_matched_noise(data[name],donor,42)
   self.assertAlmostEqual(float(torch.linalg.vector_norm((donor-data[name]).double())),float(torch.linalg.vector_norm((control-data[name]).double())),places=6)
   self.assertEqual(len(dict(m.named_modules())[name]._forward_hooks),0)

 def test_unrelated_donor_requires_same_language_length_and_different_pair(self):
  from activation_patching import choose_cases
  from test_probe_inference import Tokenizer
  def row(id,lang,pair):return {'id':id,'pair_id':pair,'task':'knowledge','input_language':lang,'output_language':lang,'prompt':'a sufficiently long prompt','answer':['x'],'scoring':{'kind':'aliases'}}
  rows=[row('a','en','first'),row('b','en','second'),row('c','sw','third')]
  cases,encoded=choose_cases(Tokenizer(),rows,4)
  for target,donor in cases:
   if target['input_language']=='sw':self.assertIsNone(donor)
   else:self.assertEqual(donor['input_language'],'en');self.assertNotEqual(donor['pair_id'],target['pair_id']);self.assertEqual(len(encoded[donor['id']]),len(encoded[target['id']]))

 def test_numeric_reference_satisfies_content_and_output_format(self):
  from activation_patching import reference_text
  from probes import score_response
  row={'answer':'42','scoring':{'kind':'number'}}
  text=reference_text(row);score=score_response(row,text);self.assertTrue(score['content_correct']);self.assertTrue(score['format_passed']);self.assertTrue(text.startswith('####'))
  row['scoring']['strict_pattern']=r'<answer>(\d+)</answer>'
  with self.assertRaises(ValueError):reference_text(row)

 def test_neutral_control_preserves_wrapper_language_and_length(self):
  from activation_patching import neutral_control_prefix
  class Tokenizer:
   def encode(self,text,**kwargs):
    if text=='z':return [9]
    if text=='.':return [10]
    return [14,15,16] if ' ni ' in text else [11,12,13]
   def apply_chat_template(self,messages,**kwargs):return [1,2,9,3,4]
  en,meta=neutral_control_prefix(Tokenizer(),'en',17);sw,_=neutral_control_prefix(Tokenizer(),'sw',17)
  self.assertEqual(len(en),17);self.assertEqual(en[:2],[1,2]);self.assertEqual(en[-2:],[3,4]);self.assertIn(11,en);self.assertIn(14,sw);self.assertTrue(meta['token_level_length_matched'])
