import unittest
import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from core import permutation, remap, score_pairs, score_labels, softmax, pride_prior, corrected
from prepare import invalid
from analyze import ci, row_metrics


class Tests(unittest.TestCase):
    def test_teacher_forcing_and_padding(self):
        torch.manual_seed(4)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=43,hidden_size=16,intermediate_size=24,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval()
        pairs=[([1,2,3],[4,5]),([2,3],[7]),([1,2,4,6],[8,9,10]),([3],[5,11,2,9])]
        with torch.inference_mode():
            scores, lengths=score_pairs(model,pairs,'cpu',0)
            expected=[]
            for c,a in pairs:
                logits=model(torch.tensor([(c+a)[:-1]])).logits[0].float().log_softmax(-1)
                expected.append(sum(float(logits[len(c)-1+i,t]) for i,t in enumerate(a)))
        np.testing.assert_allclose(scores,expected,atol=2e-6,rtol=0)
        self.assertEqual(lengths,[2,1,3,4])

    def test_shared_label_prefix(self):
        torch.manual_seed(5)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=43,hidden_size=16,intermediate_size=24,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval()
        cases=[[([1,2,3],[6,i]) for i in [7,8,9,10]], [([1,2],[i]) for i in [7,8,9,10]]]
        with torch.inference_mode():
            out=score_labels(model,cases,'cpu',0)
            for got,pairs in zip(out,cases):
                want,_=score_pairs(model,pairs,'cpu',0)
                np.testing.assert_allclose(got,want,atol=2e-6,rtol=0)

    def test_rotation_identity_and_pride(self):
        bias=np.array([1.,-1.,.3,-.3]); content=np.array([.4,1.3,-.7,.2]); scores=[]
        for r in range(4):
            order=permutation(r);self.assertEqual(sorted(order),list(range(4)))
            scores.append(bias+content[order])
            np.testing.assert_allclose(remap(content[order],order),content)
        prior=pride_prior([scores])
        np.testing.assert_allclose(prior,softmax(bias),atol=1e-12)
        for r,s in enumerate(scores):
            self.assertEqual(permutation(r)[int(corrected(s,prior).argmax())],int(content.argmax()))

    def test_rotatable_options(self):
        self.assertEqual(invalid(['Both A and B','x','y','z']),'position_dependent_option')
        self.assertEqual(invalid(['X',' x ','y','z']),'duplicate_options')
        self.assertIsNone(invalid(['alpha','beta','gamma','delta']))

    def test_cluster_weighting(self):
        result=ci([1,1,0],['a','a','b'])
        self.assertAlmostEqual(result['mean'],2/3)
        self.assertEqual(result['clusters'],2)
        self.assertEqual((result['low'],result['high']),(0.,1.))

    def test_uniform_ensemble_is_not_semantic_recovery(self):
        row=dict(id='x',cluster='x',subject='x',gold=2,rotations=[
            dict(rotation=r,order=permutation(r),letter_lp=[1.,0.,0.,0.],
                 content_lp=[0.,0.,0.,0.],content_mean_lp=[0.,0.,0.,0.],content_tokens=[1,1,1,1]) for r in range(4)])
        value=row_metrics(row,np.ones(4)/4,np.ones(4)/4)
        self.assertEqual(value['position_average_tie_count'],4)
        self.assertEqual(value['position_average_fractional_tie_correct'],.25)
        self.assertEqual(value['letter_unanimity'],0)
        self.assertEqual(value['label_unanimity'],1)

if __name__=='__main__':unittest.main()
