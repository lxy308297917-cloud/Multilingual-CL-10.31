import copy,unittest,torch
from test_weight_delta import model
from module_swap import resolve_plan
class PlanTests(unittest.TestCase):
 def test_random_parameter_and_norm_match(self):
  a=model();b=copy.deepcopy(a)
  with torch.no_grad():
   for i,layer in enumerate(b.model.layers):
    for p in layer.self_attn.parameters():p.add_(.01*(i+1))
  plan={'conditions':[{'id':'ref','selector':{'layers':[1],'component':'attention'},'mode':'restore','alpha':0},{'id':'control','selector':{'random_like':'ref','seed':42},'norm_like':'ref','mode':'restore','alpha':0}]}
  r=resolve_plan(a,b,plan);self.assertEqual(r['ref']['parameter_count'],r['control']['parameter_count']);self.assertNotEqual(r['control']['layers'],[1]);self.assertEqual(r['control']['matched_norm'],r['ref']['update_norm'])
  self.assertEqual(r['control']['layers'],resolve_plan(a,b,plan)['control']['layers'])
 def test_duplicate_label_rejected(self):
  a=model();item={'id':'x','selector':{'layers':[0],'component':'block'},'mode':'inject','alpha':1}
  with self.assertRaises(ValueError):resolve_plan(a,copy.deepcopy(a),{'conditions':[item,item]})
