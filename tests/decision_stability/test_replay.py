import torch,unittest
from losses_replay import full_vocabulary_replay
class ReplayTests(unittest.TestCase):
 def test_non_candidate_logits_receive_gradients(self):
  x=torch.zeros(4,10,requires_grad=True);loss=full_vocabulary_replay(x,[0,1,2,3]);loss.backward()
  self.assertAlmostEqual(loss.item(),torch.tensor(10.).log().item(),places=6)
  self.assertTrue(bool((x.grad[:,4:]>0).all()))
 def test_uniform_non_candidate_mass_changes_full_replay(self):
  x=torch.zeros(4,10);y=x.clone();y[:,4:]=5
  self.assertGreater(full_vocabulary_replay(y,[0,1,2,3]).item(),full_vocabulary_replay(x,[0,1,2,3]).item()+3)
if __name__=='__main__':unittest.main()
