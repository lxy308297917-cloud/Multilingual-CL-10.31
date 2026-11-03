import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/next_stage'))
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from metrics import option_log_probabilities,summarize_options,shifted_assistant_nll
class Metrics(unittest.TestCase):
 def test_shifted_mask_and_token_weighting(self):
  torch.manual_seed(7);logits=torch.randn(2,5,11);labels=torch.tensor([[-100,-100,2,3,4],[-100,8,-100,-100,-100]])
  loss,n=shifted_assistant_nll(logits,labels);oracle=sum(float(-logits[b,t-1].log_softmax(-1)[labels[b,t]]) for b in range(2) for t in range(1,5) if labels[b,t]!=-100);self.assertEqual(n,4);self.assertAlmostEqual(loss,oracle,places=5)
 def test_option_summary_and_margin(self):
  s=summarize_options([-5,-2,-3,-4],2);self.assertEqual(s['predicted'],1);self.assertFalse(s['correct']);self.assertEqual(s['gold_minus_best_wrong_margin'],-1);self.assertAlmostEqual(s['correct_answer_4option_log_probability'],float(torch.tensor([-5.,-2,-3,-4]).log_softmax(0)[2]),places=6)
 def test_actual_qwen_common_prefix_and_fallback(self):
  torch.manual_seed(42);m=Qwen3ForCausalLM(Qwen3Config(vocab_size=37,hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,attention_dropout=0.0));m.eval()
  for cs in [[[8],[9],[10],[11]],[[6,8],[6,9],[6,10],[6,11]],[[7,8],[6,9,10],[10],[11,12]]]:
   with torch.inference_mode():
    got=option_log_probabilities(m,[1,2,3],cs,'cpu');expected=[]
    for c in cs:
     seq=torch.tensor([[1,2,3]+c]);log=m(seq[:,:-1],use_cache=False).logits[0].log_softmax(-1);expected.append(sum(float(log[2+i,t]) for i,t in enumerate(c)))
   self.assertTrue(torch.allclose(torch.tensor(got),torch.tensor(expected),atol=1e-6,rtol=1e-6),(got,expected))
if __name__=='__main__':unittest.main()
