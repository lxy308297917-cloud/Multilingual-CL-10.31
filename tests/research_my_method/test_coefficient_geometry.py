import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/research_my_method"))
import sys,unittest,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/sst'))
from projector import Subspace
from coefficient_geometry import compact_update_modes,centered_cross_space,project_coefficient
class Tests(unittest.TestCase):
 def test_full_rectangular_space_and_reconstruction(self):
  torch.manual_seed(4)
  for m,n in [(12,12),(21,12),(12,21)]:
   w=torch.randn(m,n,dtype=torch.float64);x=torch.randn(m,n,dtype=torch.float64);b=Subspace.from_weight(w,.7);u,s,v=compact_update_modes(x,b);rebuild=(u.double()*s.double())@v.double().T;torch.testing.assert_close(rebuild,b.project(x),rtol=2e-6,atol=2e-6);torch.testing.assert_close((u.T@u)*(v.T@v),torch.eye(len(s)),rtol=2e-6,atol=2e-6)
 def test_pairing_is_not_lost_by_averaging(self):
  g=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]);q,vals=centered_cross_space(g,g,['a']*4,max_rank=2);self.assertEqual(q.shape[1],2)
  shuffled=g[[1,0,3,2]];qq,vv=centered_cross_space(g,shuffled,['a']*4,max_rank=2);self.assertEqual(qq.shape[1],0);self.assertTrue(torch.equal(g.mean(0),shuffled.mean(0)))
  d=torch.tensor([2.,3.]);q=torch.tensor([[1.],[0.]]);torch.testing.assert_close(project_coefficient(d,q),torch.tensor([0.,3.]));torch.testing.assert_close(project_coefficient(d,q,True),torch.tensor([2.,0.]))
if __name__=='__main__':unittest.main()
