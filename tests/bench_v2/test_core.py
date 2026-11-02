import unittest,tempfile,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/bench_v2'))
from common import PredictionStore,require_complete,checked_rows,dump,object_hash,select_ids
from prompting import render
from scoring import mt_metrics,paired_bootstrap,repetition

class Tokenizer:
 def encode(self,text,add_special_tokens=False):return list(map(ord,text))
 def decode(self,ids,skip_special_tokens=False):return ''.join(map(chr,ids))
 def apply_chat_template(self,messages,tokenize,add_generation_prompt,enable_thinking):
  if enable_thinking:raise AssertionError('Thinking must be disabled')
  text=''.join('<'+r['role']+'>'+r['content'] for r in messages)+'<assistant>'
  return self.encode(text) if tokenize else text

class Records(unittest.TestCase):
 def test_resume_skips_completed_and_rejects_changed_identity(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'run';s=PredictionStore(out,{'method':'base','protocol':'v2'},['a','b']);s.append([{'id':'a','prediction':'x'}]);self.assertEqual(s.finish('full')['run_status'],'PARTIAL');r=PredictionStore(out,{'method':'base','protocol':'v2'},['a','b'],resume=True);self.assertEqual(r.pending(),['b']);r.append([{'id':'b','prediction':'y'}]);self.assertEqual(r.finish('full')['run_status'],'VERIFIED_COMPLETE')
   with self.assertRaises(ValueError):PredictionStore(out,{'method':'full','protocol':'v2'},['a','b'],resume=True)
 def test_pilot_is_not_formal(self):
  with tempfile.TemporaryDirectory() as d:
   s=PredictionStore(Path(d)/'x',{},['a']);s.append([{'id':'a'}]);self.assertEqual(s.finish('pilot')['run_status'],'PILOT_ONLY')
 def test_duplicates_are_errors(self):
  with tempfile.TemporaryDirectory() as d:
   s=PredictionStore(Path(d)/'x',{},['a'])
   with self.assertRaises(ValueError):s.append([{'id':'a'},{'id':'a'}])
   self.assertFalse(s.rows)
 def test_incomplete_cannot_be_scored(self):
  with self.assertRaises(ValueError):require_complete([{'id':'a'}],['a','b'])
  with self.assertRaises(ValueError):require_complete([{'id':'a'},{'id':'a'}],['a'])
 def test_dataset_ids_and_expected_counts(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'test.jsonl';p.write_text('{"id":"a"}\n{"id":"a"}\n')
   with self.assertRaises(ValueError):checked_rows(p)
   p.write_text('{"id":"a"}\n')
   with self.assertRaises(ValueError):checked_rows(p,2)
 def test_pilot_ids_are_score_independent(self):
  rows=[{'id':str(i)} for i in range(30)];self.assertEqual(select_ids(rows,42,8),select_ids(list(reversed(rows)),42,8));self.assertEqual(len(select_ids(rows,42,8)),8)

class Prompts(unittest.TestCase):
 def test_article_only_head_truncation(self):
  t=Tokenizer();r=render(t,{'source':'abcd'*100,'article_prompt_template':'SW SUMMARY: {article}'},80);self.assertTrue(r['article_truncated']);self.assertLessEqual(r['input_tokens'],80);self.assertIn('SW SUMMARY:',r['prompt']);self.assertEqual(r['article_tokens_before_truncation'],400)
 def test_no_silent_truncation_other_tasks(self):
  with self.assertRaises(ValueError):render(Tokenizer(),{'messages':[{'role':'user','content':'x'*100}]},20)
 def test_identical_render_for_every_method(self):
  row={'messages':[{'role':'user','content':'Tafsiri hili.'}]};a=render(Tokenizer(),row,80);b=render(Tokenizer(),row,80);self.assertEqual(a,b);self.assertEqual(object_hash(a['prompt_token_ids']),object_hash(b['prompt_token_ids']))

class Metrics(unittest.TestCase):
 def test_exact_match_standard_metrics(self):
  score,signature=mt_metrics(['this is a sufficiently long test sentence'],['this is a sufficiently long test sentence']);self.assertAlmostEqual(score['chrF++'],100);self.assertAlmostEqual(score['BLEU'],100);self.assertIn('nw:2',signature['chrF++'])
 def test_mismatched_metric_inputs_rejected(self):
  with self.assertRaises(ValueError):mt_metrics(['x'],[])
 def test_paired_identical_generations_have_zero_delta(self):
  r=paired_bootstrap(['a b c d','w x y z'],['a b c d','w x y z'],['a b c d','q r s t'],repeats=12);self.assertEqual(r['delta'],0);self.assertEqual(r['ci95_low'],0);self.assertEqual(r['ci95_high'],0)
 def test_repetition_is_explicit_rule(self):
  self.assertTrue(repetition('hello world '*10));self.assertFalse(repetition('hello world '*9));self.assertFalse(repetition('normal unique translated sentence'))
if __name__=='__main__':unittest.main()
