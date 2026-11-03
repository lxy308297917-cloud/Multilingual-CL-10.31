import copy,io,sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/sst'))
from projector import Subspace,AnchoredState
from optimizer_adapter import TorchSST

torch.set_num_threads(1)
class ProjectionTests(unittest.TestCase):
 def test_explicit_rectangular_and_extreme_ranks(self):
  torch.manual_seed(42)
  for dtype in [torch.float64,torch.float32]:
   for m,n in [(9,9),(5,19),(21,6),(1,8),(8,1)]:
    for ratio in [0.,.01,.5,.99,1.]:
     with self.subTest(dtype=dtype,shape=(m,n),ratio=ratio):
      w=torch.randn(m,n,dtype=dtype);d=torch.randn_like(w);b=Subspace.from_weight(w,ratio);u,_,vh=torch.linalg.svd(w,full_matrices=False);k=b.rank
      explicit=(torch.eye(m,dtype=dtype)-u[:,:k]@u[:,:k].T)@d@(torch.eye(n,dtype=dtype)-vh[:k].T@vh[:k]);actual=b.project(d);tol=1e-11 if dtype==torch.float64 else 2e-5
      torch.testing.assert_close(actual,explicit,atol=tol,rtol=tol)
      for x in [u[:,:k].T@actual,actual@vh[:k].T]:self.assertLess(float(x.norm()),tol*max(1.,float(d.norm())))
      self.assertEqual(b.inventory()['free_space_dimension'],(m-k)*(n-k))
 def test_rectangular_nullspace_is_kept(self):
  torch.manual_seed(11);w=torch.randn(15,4,dtype=torch.float64);u,_,v=torch.linalg.svd(w,full_matrices=True);b=Subspace.from_weight(w,.99)
  d=u[:,5:6]@v[-1:];torch.testing.assert_close(b.project(d),d,atol=1e-12,rtol=1e-12)
 def test_invalid_inputs(self):
  for ratio in [-1,1.1]:
   with self.assertRaises(ValueError):Subspace.from_weight(torch.eye(3),ratio)
  with self.assertRaises(TypeError):Subspace.from_weight(torch.eye(3,dtype=torch.bfloat16))

class OptimizerTests(unittest.TestCase):
 def test_full_adam_moments_and_gradient_projection_negative_control(self):
  torch.manual_seed(7);w=torch.randn(7,5,dtype=torch.float64);p=torch.nn.Parameter(w.clone());q=torch.nn.Parameter(w.clone());r=torch.nn.Parameter(w.clone());a=torch.optim.AdamW([p],lr=.01,weight_decay=.03);b=torch.optim.AdamW([q],lr=.01,weight_decay=.03);c=torch.optim.AdamW([r],lr=.01,weight_decay=.03);sst=TorchSST(a,{'w':p},.8);basis=sst.constraint.bases['w']
  for _ in range(4):
   g=torch.randn_like(w);p.grad=g.clone();q.grad=g.clone();r.grad=basis.project(g);before=p.detach().clone();_,raw=sst.step();b.step();c.step();delta=p.detach()-before
   for key in ['exp_avg','exp_avg_sq','step']:torch.testing.assert_close(a.state[p][key],b.state[q][key],rtol=0,atol=0)
   self.assertLess(basis.leakage(delta)['left_relative'],1e-10);self.assertLess(basis.leakage(delta)['right_relative'],1e-10)
   self.assertGreater(float((raw['w']-delta).norm()),1e-5)
  self.assertGreater(float((p-r).norm()),1e-3)
  self.assertGreater(basis.leakage(r.detach()-w)['left_relative'],.1)
 def test_cumulative_and_incremental_equivalence(self):
  torch.manual_seed(2);w=torch.randn(8,5,dtype=torch.float64);s=AnchoredState({'w':w},.8);x=w.clone();y=w.clone()
  for _ in range(30):
   d=torch.randn_like(w)*.02;x=s.project_proposal({'w':x+d})['w'];y=y+s.bases['w'].project(d)
   torch.testing.assert_close(x,y,atol=2e-14,rtol=2e-14)
   self.assertLess(s.bases['w'].leakage(x-w)['left_relative'],1e-11)
 def test_refresh_uses_projected_weight_at_ten_and_twenty(self):
  torch.manual_seed(17);x=torch.randn(8,5,dtype=torch.float64);s=AnchoredState({'w':x},.8,10)
  for step in range(1,22):
   projected=s.project_proposal({'w':x+torch.randn_like(x)*.01});x=projected['w'];old=s.references['w'];s.completed_step(projected)
   self.assertEqual(s.step,step);self.assertEqual(s.refresh_steps,[v for v in [10,20] if v<=step])
   if step in [10,20]:self.assertIsNot(old,s.references['w']);torch.testing.assert_close(s.references['w'],x,rtol=0,atol=0)
   else:self.assertIs(old,s.references['w'])
 def test_resume_matches_parameters_optimizer_scheduler_rng_and_basis(self):
  for interval in [None,10]:
   for stop in [9,10,17]:
    with self.subTest(interval=interval,stop=stop):
     def build():
      p=torch.nn.Parameter(torch.randn(7,5,dtype=torch.float64));opt=torch.optim.AdamW([p],lr=.01,weight_decay=0);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=25);return p,opt,sched,TorchSST(opt,{'w':p},.8,interval)
     torch.manual_seed(101);p,opt,sched,sst=build();snapshot=None
     for step in range(25):
      p.grad=torch.randn_like(p);sst.step();sched.step()
      if step+1==stop:
       snapshot=copy.deepcopy({'p':p.detach(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'sst':sst.constraint.state_dict(),'rng':torch.get_rng_state()})
     expected=p.detach().clone();expected_state=copy.deepcopy(opt.state_dict());expected_rng=torch.get_rng_state();q,oq,sq,r=build();q.data.copy_(snapshot['p']);oq.load_state_dict(snapshot['optimizer']);sq.load_state_dict(snapshot['scheduler']);r.constraint=AnchoredState.from_state_dict(snapshot['sst']);torch.set_rng_state(snapshot['rng'])
     for _ in range(stop,25):q.grad=torch.randn_like(q);r.step();sq.step()
     torch.testing.assert_close(q,expected,rtol=0,atol=0);self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng));self.assertEqual(sq.state_dict(),sched.state_dict());self.assertEqual(r.constraint.refresh_steps,sst.constraint.refresh_steps)
     for key,value in expected_state['state'][0].items():torch.testing.assert_close(oq.state_dict()['state'][0][key],value,rtol=0,atol=0)
     torch.testing.assert_close(r.constraint.references['w'],sst.constraint.references['w'],rtol=0,atol=0)
     for name in ['left','right']:torch.testing.assert_close(getattr(r.constraint.bases['w'],name),getattr(sst.constraint.bases['w'],name),rtol=0,atol=0)
 def test_identity_matches_unmodified_adam(self):
  torch.manual_seed(8);p=torch.nn.Parameter(torch.randn(8,5));q=torch.nn.Parameter(p.detach().clone());a=torch.optim.AdamW([p],lr=.01);b=torch.optim.AdamW([q],lr=.01);sst=TorchSST(a,{'w':p},0)
  for _ in range(15):p.grad=torch.randn_like(p);q.grad=p.grad.clone();sst.step();b.step();torch.testing.assert_close(p,q,rtol=0,atol=0)
if __name__=='__main__':unittest.main(verbosity=2)
