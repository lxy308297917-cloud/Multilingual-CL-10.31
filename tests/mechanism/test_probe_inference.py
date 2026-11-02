import unittest
from test_weight_delta import model
from inference import evaluate_rows,summarize,encode_probe
class Tokenizer:
 bos_token_id=1;eos_token_id=2;pad_token_id=0
 def apply_chat_template(self,messages,**kwargs):
  assert kwargs['enable_thinking'] is False
  return [1]+[3+ord(c)%50 for c in messages[0]['content'][:16]]
 def encode(self,text,**kwargs):return [5+'ABCD'.index(text)]
 def decode(self,ids,**kwargs):return ' '.join(str(x) for x in ids if x not in [0,1,2])
class InferenceTests(unittest.TestCase):
 def test_fixed_protocol_reproducible(self):
  m=model();t=Tokenizer();rows=[{'id':'a','pair_id':'p','task':'knowledge','input_language':'en','output_language':'sw','prompt':'Which answer?','choices':['x','y','z','w'],'answer':'A','scoring':{'kind':'choice'},'source':{'role':'synthetic_smoke'},'quality':{'status':'toy'}},{'id':'b','pair_id':'q','task':'math','input_language':'sw','output_language':'en','prompt':'1+1?','answer':'2','scoring':{'kind':'number'},'source':{'role':'synthetic_smoke'},'quality':{'status':'toy'}}]
  a=evaluate_rows(m,t,rows,max_new_tokens=2);b=evaluate_rows(m,t,rows,max_new_tokens=2)
  self.assertEqual(a,b);self.assertEqual(len(a[0]['score']['option_scores']),4);self.assertEqual(len(summarize(a)),2)
  self.assertIsNone(a[0]['score']['language_correct'])
 def test_no_silent_truncation(self):
  with self.assertRaises(ValueError):encode_probe(Tokenizer(),{'prompt':'long prompt','scoring':{'kind':'number'},'input_language':'en','output_language':'en'},max_input_tokens=1)

 def test_sampling_model_defaults_do_not_override_greedy(self):
  from transformers import GenerationConfig
  m=model();m.generation_config.do_sample=True;m.generation_config.temperature=.6;m.generation_config.top_p=.95
  cfg=GenerationConfig(do_sample=False,eos_token_id=2,pad_token_id=0,max_new_tokens=3)
  effective,_=m._prepare_generation_config(cfg,use_model_defaults=False)
  self.assertFalse(effective.do_sample);self.assertEqual(effective.temperature,1.);self.assertEqual(effective.top_p,1.)
  row={'id':'x','pair_id':'p','task':'math','input_language':'en','output_language':'en','prompt':'2+2?','answer':'4','scoring':{'kind':'number'},'source':{'role':'synthetic_smoke'},'quality':{'status':'toy'}}
  self.assertEqual(evaluate_rows(m,Tokenizer(),[row],max_new_tokens=3),evaluate_rows(m,Tokenizer(),[row],max_new_tokens=3))
