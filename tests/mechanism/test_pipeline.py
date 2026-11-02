import json,os,subprocess,tempfile,unittest
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from test_weight_delta import model
class PipelineTests(unittest.TestCase):
 def test_cpu_local_tiny_full_end_to_end(self):
  import sys
  root=Path(__file__).resolve().parents[2]
  with tempfile.TemporaryDirectory() as directory:
   d=Path(directory);base=d/'base';final=d/'final';m=model();m.save_pretrained(base)
   with torch.no_grad():
    for p in m.model.layers[1].self_attn.parameters():p.add_(.001)
   m.save_pretrained(final)
   vocabulary={'[UNK]':0,'[PAD]':1,'[EOS]':2,'A':3,'B':4,'C':5,'D':6,'user':7,'assistant':8}
   tokenizer=Tokenizer(WordLevel(vocabulary,unk_token='[UNK]'));tokenizer.pre_tokenizer=Whitespace();t=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]');t.chat_template="{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}";t.save_pretrained(base)
   row={'id':'one','pair_id':'p','task':'knowledge_choice','input_language':'en','output_language':'en','prompt':'Which?','answer':'A','choices':['one','two','three','four'],'scoring':{'kind':'choice'},'source':{'dataset':'toy','version':'v1','split':'constructed','original_id':'x','role':'synthetic_smoke'},'quality':{'status':'toy','parallel_verified':False}}
   probes=d/'probes.jsonl';probes.write_text(json.dumps(row)+'\n');plan=d/'plan.json';plan.write_text(json.dumps({'formal_test_used_for_selection':False,'max_new_tokens':2,'max_input_tokens':128,'conditions':[{'id':'identity','selector':{'layers':[1],'component':'attention'},'mode':'restore','alpha':1},{'id':'restore','selector':{'layers':[1],'component':'attention'},'mode':'restore','alpha':0},{'id':'inject','selector':{'layers':[1],'component':'attention'},'mode':'inject','alpha':1}]}))
   result=subprocess.run([sys.executable,str(root/'scripts/mechanism/module_swap.py'),'--base',str(base),'--final',str(final),'--kind','full','--probes',str(probes),'--plan',str(plan),'--output',str(d/'results')],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'})
   self.assertEqual(result.returncode,0,result.stdout+result.stderr)
   done=json.loads((d/'results/DONE.json').read_text());self.assertEqual(done['conditions'],['base','final','identity','restore','inject'])
   ref=(d/'results/final/predictions.jsonl').read_text();self.assertEqual(ref,(d/'results/identity/predictions.jsonl').read_text())
   for label in ['identity','restore','inject']:
    metadata=json.loads((d/'results'/label/'summary.json').read_text())['metadata'];self.assertTrue(metadata['intervention']['restoration_verified'])

   activation=subprocess.run([sys.executable,str(root/'scripts/mechanism/audit_activation_drift.py'),'--base',str(base),'--final',str(final),'--kind','full','--probes',str(probes),'--output',str(d/'activation')],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'})
   self.assertEqual(activation.returncode,0,activation.stdout+activation.stderr)
   activation_report=json.loads((d/'activation/activation_drift.json').read_text());self.assertEqual(activation_report['sample_count'],1);self.assertEqual(len(activation_report['rows']),18)

   functional=subprocess.run([sys.executable,str(root/'scripts/mechanism/activation_patching.py'),'--base',str(base),'--final',str(final),'--kind','full','--probes',str(probes),'--output',str(d/'patching'),'--layer','1','--max-per-group','1'],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'})
   self.assertEqual(functional.returncode,0,functional.stdout+functional.stderr)
   patch_summary=json.loads((d/'patching/DONE.json').read_text());self.assertEqual(patch_summary['samples'],1);self.assertEqual(patch_summary['records'],12)
