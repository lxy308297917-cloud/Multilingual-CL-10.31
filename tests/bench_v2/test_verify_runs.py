import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/bench_v2'))
import copy, unittest
from verify_runs import verify_group
class ManifestVerification(unittest.TestCase):
 def fixture(self):
  b=dict(method='base',scope='full',sample_ids=['a'],sample_ids_sha256='a',benchmark_id='x',data_manifest_hash='d',tokenizer='base',prompt_tokens_sha256='p',backend={'dtype':'bfloat16'},batch_size=32,source_files_sha256={'generate.py':'code'},benchmark_protocol={'enable_thinking':False},checkpoint_identity={'base/model.safetensors':'w'},training_seed=42)
  g={k:dict(copy.deepcopy(b),method=k) for k in ['base','full','lora','ssu']};g['lora']['checkpoint_identity']['adapter/adapter_model.safetensors']='a';return g
 def test_aligned(self):self.assertTrue(verify_group(self.fixture(),1))
 def test_drift_rejected(self):
  for field,value in [('prompt_tokens_sha256','changed'),('tokenizer','other'),('batch_size',4)]:
   g=self.fixture();g['ssu'][field]=value
   with self.assertRaises(ValueError):verify_group(g,1)
 def test_missing_adapter_rejected(self):
  g=self.fixture();del g['lora']['checkpoint_identity']['adapter/adapter_model.safetensors']
  with self.assertRaises(ValueError):verify_group(g,1)
if __name__=='__main__':unittest.main()
