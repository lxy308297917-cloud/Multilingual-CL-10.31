"""Small independent oracles for outcome-critical scoring edge cases."""
from runtime import runtime_path
import unittest
from common import number, numeric_prediction
from evaluate import tree, metrics

class Scoring(unittest.TestCase):

    def test_number(self):
        self.assertEqual(number('Therefore #### -1,234.50'), number('-1234.5'))
        self.assertEqual(number('Résultat : 12,5 euros', 'fr'), number('12.5'))
        self.assertIsNone(number('no answer'))
        self.assertIsNone(numeric_prediction('the answer is 42'))
        self.assertEqual(numeric_prediction('#### 42'), number('42'))

    def test_tree(self):
        self.assertEqual(tree('[IN:CREATE_CALL [SL:CONTACT Jane Doe ] ]'), tree('[IN:CREATE_CALL\n[SL:CONTACT Jane   Doe]]'))
        self.assertNotEqual(tree('[IN:CREATE_CALL [SL:CONTACT Jane Doe]]'), tree('[IN:CREATE_CALL [SL:CONTACT Doe Jane]]'))
        for bad in ['blah [IN:CREATE_CALL]', '[IN:CREATE_CALL', '[SL:CONTACT Jane]', '[IN:CREATE_CALL] trailing']:
            with self.assertRaises(ValueError):
                tree(bad)

    def test_label(self):
        rs = [dict(track='massive', output='alarm_set', answer='alarm_set', labels=['alarm_set', 'alarm_remove'], finish_reason='stop'), dict(track='massive', output='The answer is alarm_set', answer='alarm_set', labels=['alarm_set', 'alarm_remove'], finish_reason='stop')]
        result = metrics(rs)
        self.assertEqual(result['score'], 0.5)
        self.assertEqual(result['invalid_label_rate'], 0.5)

    def test_math_format_separation(self):
        rs = [dict(track='mathoctopus', lang='en', output='The answer is 42.', gold_numeric=number('42'), finish_reason='stop')]
        m = metrics(rs)
        self.assertEqual(m['score'], 1)
        self.assertEqual(m['strict_accuracy'], 0)
if __name__ == '__main__':
    unittest.main()
