import unittest,tempfile,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/mechanism'))
from probes import score_response,load_probes
class ProbeTests(unittest.TestCase):
 def test_math_content_format_separate(self):
  row={'answer':'42','scoring':{'kind':'number'}}
  self.assertEqual(score_response(row,'<answer>42</answer>')['content_correct'],True)
  self.assertEqual(score_response(row,'<answer>42</answer>')['format_passed'],False)
  self.assertEqual(score_response(row,'#### 42')['format_passed'],True)
  self.assertFalse(score_response(row,'I used 42 apples')['content_correct'])
 def test_choice_uses_scores(self):
  row={'answer':'B','scoring':{'kind':'choice'}}
  self.assertTrue(score_response(row,'A',[0.,1.,-.5,-2.])['content_correct'])
  with self.assertRaises(ValueError):score_response(row,'B',[0,float('nan'),1,2])
 def test_instruction_no_fake_content(self):
  row={'answer':None,'scoring':{'kind':'constraints','constraints':[{'kind':'exact_words','value':2},{'kind':'contains','value':'sawa'}]}}
  r=score_response(row,'sawa kabisa');self.assertTrue(r['format_passed']);self.assertIsNone(r['content_correct']);self.assertIsNone(r['language_correct'])
 def test_test_split_rejected(self):
  row={'id':'1','pair_id':'x','task':'math','input_language':'en','output_language':'en','prompt':'1+1?','answer':'2','scoring':{'kind':'number'},'source':{'dataset':'anything','version':'v1','split':'test','original_id':'1','role':'independent_development'},'quality':{'status':'reviewed','parallel_verified':True}}
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'p.jsonl';p.write_text(json.dumps(row)+'\n')
   with self.assertRaises(ValueError):load_probes(p)
   row['source']['split']='constructed';row['source']['role']='synthetic_smoke';p.write_text(json.dumps(row)+'\n');self.assertEqual(len(load_probes(p)),1)

 def test_unicode_line_separator_inside_json_string(self):
  row={'id':'unicode','pair_id':None,'task':'reading','input_language':'sw','output_language':'sw','prompt':'line one\u2028line two','answer':['two'],'scoring':{'kind':'aliases'},'source':{'dataset':'original','version':'v1','split':'constructed','original_id':'x','role':'synthetic_smoke'},'quality':{'status':'toy','parallel_verified':False}}
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'p.jsonl';p.write_text(json.dumps(row,ensure_ascii=False)+'\n');self.assertEqual(load_probes(p)[0]['prompt'],row['prompt'])
