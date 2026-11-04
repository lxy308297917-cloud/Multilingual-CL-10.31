import unittest
import torch
from losses import auxiliary_loss,content_average
class LossTests(unittest.TestCase):
    def setUp(self):
        self.orders=[[(j+r)%4 for j in range(4)] for r in range(4)]
    def test_content_identity(self):
        target=torch.tensor([.1,.2,.3,.4])
        logits=torch.stack([target[o].log() for o in self.orders]).requires_grad_()
        torch.testing.assert_close(content_average(logits,self.orders),target)
        loss=auxiliary_loss('content',logits,self.orders,3,target,[.25]*4)
        self.assertLess(abs(loss.item()),1e-6);loss.backward();self.assertLess(logits.grad.abs().max(),1e-6)
    def test_marginal_not_semantics(self):
        target=torch.tensor([.1,.2,.3,.4]);logits=torch.stack([target[o].log() for o in self.orders]).requires_grad_()
        marginal=auxiliary_loss('marginal',logits,self.orders,3,target,[.25]*4)
        self.assertLess(abs(marginal.item()),1e-6)
        changed=auxiliary_loss('content',logits,self.orders,3,target.roll(1),[.25]*4)
        self.assertGreater(changed.item(),.05)
    def test_replay_matches_four_ce(self):
        x=torch.randn(4,4,requires_grad=True)
        got=auxiliary_loss('replay',x,self.orders,2,[.25]*4,[.25]*4)
        want=sum(-x[i].log_softmax(0)[o.index(2)] for i,o in enumerate(self.orders))/4
        torch.testing.assert_close(got,want)
        g1=torch.autograd.grad(got,x,retain_graph=True)[0];g2=torch.autograd.grad(want,x)[0];torch.testing.assert_close(g1,g2)
    def test_accumulation_once(self):
        p=torch.tensor(2.,requires_grad=True)
        summed=sum(((p-1)**2+(8*(p+1)**2 if i==0 else 0))/8 for i in range(8))
        expected=(p-1)**2+(p+1)**2
        torch.testing.assert_close(torch.autograd.grad(summed,p,retain_graph=True)[0],torch.autograd.grad(expected,p)[0])
if __name__=='__main__':unittest.main()
