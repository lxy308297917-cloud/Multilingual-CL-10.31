import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/mechanism'))
from diagnose_math_answers import resolve_numeric_response
class NumericDiagnosticTests(unittest.TestCase):
 def test_short_unmarked_answer_and_wrong_answer(self):
  self.assertTrue(resolve_numeric_response('Sammy has 8 bottle caps.','8',True)['correct'])
  self.assertFalse(resolve_numeric_response('Adam has 11 tattoos.','23',True)['correct'])
 def test_no_arbitrary_last_number_rescue(self):
  self.assertFalse(resolve_numeric_response('There are 5 groups and 8 objects.','8',True)['resolved'])
  self.assertFalse(resolve_numeric_response('Sammy has 8 bottle caps.','8',False)['resolved'])
 def test_comma_explicit_answer(self):
  self.assertTrue(resolve_numeric_response('The result is:\n#### 1,000','1000',False)['correct'])

 def test_content_marker_is_separate_from_strict_end_format(self):
  self.assertTrue(resolve_numeric_response('#### 42\nDone.','42',True)['correct'])
