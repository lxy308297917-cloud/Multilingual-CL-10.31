import copy,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/mechanism'))
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from weights import drift_rows,validate_pair

def model():
 torch.manual_seed(42)
 return Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=16,intermediate_size=24,num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64,tie_word_embeddings=True)).eval()
class WeightTests(unittest.TestCase):
 def test_identity_and_change(self):
  a=model();b=copy.deepcopy(a);self.assertTrue(all(r['changed_elements']==0 for r in drift_rows(a,b)))
  with torch.no_grad():b.model.layers[0].mlp.up_proj.weight.add_(.1)
  rows=drift_rows(a,b);changed=[r for r in rows if r['changed_elements']];self.assertEqual(len(changed),1);self.assertEqual(changed[0]['changed_elements'],384)
 def test_dtype_rejected(self):
  a=model();b=copy.deepcopy(a).double()
  with self.assertRaises(ValueError):validate_pair(a,b)
 def test_lora_merge(self):
  from peft import LoraConfig,get_peft_model
  a=model();p=get_peft_model(copy.deepcopy(a),LoraConfig(r=2,lora_alpha=6,target_modules=['q_proj'],task_type='CAUSAL_LM'))
  module=p.base_model.model.model.layers[0].self_attn.q_proj
  with torch.no_grad():module.lora_A.default.weight.fill_(.2);module.lora_B.default.weight.fill_(.3)
  expected=module.get_delta_weight('default').detach().clone();merged=p.merge_and_unload(safe_merge=True)
  self.assertTrue(torch.allclose(merged.model.layers[0].self_attn.q_proj.weight-a.model.layers[0].self_attn.q_proj.weight,expected,atol=1e-7))
  validate_pair(a,merged)
