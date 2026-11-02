import copy,unittest,torch
from test_weight_delta import model
from weights import intervene,select_parameters,digest_tensor,parameters
class SwapTests(unittest.TestCase):
 def test_identity_restore_inject_and_rollback(self):
  a=model();b=copy.deepcopy(a)
  with torch.no_grad():
   for p in b.parameters():p.add_(.01)
  chosen=select_parameters(b,[1],'attention');before={n:digest_tensor(p) for n,p in parameters(b).items()}
  for mode,alpha in [('restore',0),('restore',.5),('restore',1),('inject',1)]:
   with intervene(a,b,chosen,mode,alpha) as report:
    for n,p in parameters(b).items():
     x=parameters(a)[n];original=x+.01
     expected=x+alpha*(original-x) if n in chosen else (x if mode=='inject' else original)
     self.assertTrue(torch.allclose(p,expected,atol=2e-8),n)
   self.assertTrue(report['restoration_verified']);self.assertEqual(before,{n:digest_tensor(p) for n,p in parameters(b).items()})
 def test_exception_rollback(self):
  a=model();b=copy.deepcopy(a);chosen=select_parameters(b,[0,1],'mlp');old=digest_tensor(b.model.layers[0].mlp.up_proj.weight)
  with self.assertRaisesRegex(RuntimeError,'intentional'):
   with intervene(a,b,chosen):raise RuntimeError('intentional')
  self.assertEqual(old,digest_tensor(b.model.layers[0].mlp.up_proj.weight))
 def test_all_granularities(self):
  a=model()
  for comp in ['block','attention','mlp','q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
   self.assertTrue(select_parameters(a,[0,1],comp))
  with self.assertRaises(ValueError):select_parameters(a,[99],'block')
 def test_ties_preserved(self):
  a=model();b=copy.deepcopy(a)
  with self.assertRaises(ValueError):
   with intervene(a,b,{'lm_head.weight'}):pass

 def test_identity_is_byte_exact_even_after_large_change(self):
  a=model();b=copy.deepcopy(a);chosen=select_parameters(b,[0],'q_proj')
  with torch.no_grad():
   a.model.layers[0].self_attn.q_proj.weight.fill_(100000.)
   b.model.layers[0].self_attn.q_proj.weight.fill_(.00001)
  before=digest_tensor(b.model.layers[0].self_attn.q_proj.weight)
  with intervene(a,b,chosen,'restore',1):
   self.assertEqual(before,digest_tensor(b.model.layers[0].self_attn.q_proj.weight))

 def test_norm_matched_restore_controls_removed_update(self):
  a=model();b=copy.deepcopy(a);chosen=select_parameters(b,[1],'q_proj')
  with torch.no_grad():b.model.layers[1].self_attn.q_proj.weight.add_(.01)
  with intervene(a,b,chosen,'restore',0,delta_norm=.07) as r:
   self.assertAlmostEqual(r['actual_replacement_norm'],.07,places=6)
  self.assertTrue(r['restoration_verified'])
