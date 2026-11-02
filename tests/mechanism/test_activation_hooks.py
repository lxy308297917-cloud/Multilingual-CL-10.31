import copy,unittest,torch
from test_weight_delta import model
from activations import capture,patch,compare_same_input,linear_cka
from inference import continuation_loglikelihood
class HookTests(unittest.TestCase):
 def test_capture_identity_patch_and_cleanup(self):
  m=model();name='model.layers.1.mlp';ids=torch.tensor([[1,2,3,4]])
  with torch.inference_mode():
   baseline=m(ids,use_cache=False).logits.clone()
   with capture(m,[name]) as data:m(ids,use_cache=False)
   with patch(m,name,data[name]):identical=m(ids,use_cache=False).logits
   self.assertTrue(torch.equal(baseline,identical))
   self.assertEqual(len(m.model.layers[1].mlp._forward_hooks),0)
   self.assertEqual(compare_same_input(data[name],data[name])['relative_l2'],0.)
 def test_error_cleanup_and_bad_alignment(self):
  m=model();name='model.layers.0.self_attn';ids=torch.tensor([[1,2,3]])
  with self.assertRaises(ValueError):
   with patch(m,name,torch.zeros(1,2,16)):m(ids,use_cache=False)
  self.assertEqual(len(m.model.layers[0].self_attn._forward_hooks),0)
  with self.assertRaisesRegex(RuntimeError,'intentional'):
   with capture(m,[name]):raise RuntimeError('intentional')
  self.assertEqual(len(m.model.layers[0].self_attn._forward_hooks),0)
 def test_partial_patch_does_not_mutate_donor(self):
  m=model();name='model.layers.0.mlp';ids=torch.tensor([[1,2,3]])
  with capture(m,[name],[2]) as donor:m(ids,use_cache=False)
  old=donor[name].clone()
  with patch(m,name,donor[name],[2]):m(ids,use_cache=False)
  self.assertTrue(torch.equal(old,donor[name]))
 def test_cka(self):
  x=torch.randn(8,4);self.assertAlmostEqual(linear_cka(x,x),1.,places=10)
  self.assertIsNone(linear_cka(torch.zeros(8,4),x))
 def test_continuation_causal_alignment(self):
  m=model();prefix=[1,2,3];answer=[4,5]
  got,parts=continuation_loglikelihood(m,prefix,answer)
  with torch.inference_mode():
   raw=m(torch.tensor([prefix+answer]),use_cache=False).logits
   expected=raw[0,2:4].float().log_softmax(-1).gather(1,torch.tensor(answer).view(-1,1)).sum()
   prefix_first=m(torch.tensor([prefix]),use_cache=False).logits[0,-1].float().log_softmax(-1)[answer[0]]
  self.assertAlmostEqual(got,float(expected),places=6);self.assertAlmostEqual(parts[0],float(prefix_first),places=6)
