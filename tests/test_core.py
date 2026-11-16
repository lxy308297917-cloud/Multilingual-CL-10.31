import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from s3_align.assets import Assets, read_jsonl
from s3_align.losses import alignment, eval_grad, old_objective
from s3_align.subspace import cca
from s3_align.trainer import make_trainer


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(18)])
    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        for block in self.layers:
            h = block(h)
        return SimpleNamespace(last_hidden_state=h)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Decoder()


def manual_states(model, row):
    h = model.model.embed(torch.tensor([row['input_ids']]))
    states = []
    for block in model.model.layers:
        h = block(h)
        states.append(h[0])
    return states


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = Model()
        self.row = dict(id='a', capability='math', input_ids=[1,2,3,4],
                        positions=[0,1,2,3], answer_prediction_slots=[1,3], body_positions=[0,2])
        self.ref = torch.randn(2,18,4)

    def test_keep_loss_and_gradients(self):
        actual = old_objective(self.model,self.row,'full_feature',self.ref,2.)
        states = manual_states(self.model,self.row)
        expected = torch.stack([((s[[1,3]]-self.ref[:,i]).square().sum() /
                                 self.ref[:,i].square().sum()) for i,s in enumerate(states)]).mean()/2
        torch.testing.assert_close(actual,expected)
        a = torch.autograd.grad(actual,list(self.model.parameters()))
        b = torch.autograd.grad(expected,list(self.model.parameters()))
        for x,y in zip(a,b): torch.testing.assert_close(x,y)
        self.assertTrue(all(len(layer._forward_hooks)==0 for layer in self.model.model.layers))

    def test_mapping_and_gradients(self):
        maps={'body.mean.sw':torch.randn(4),'T2.projection.sw':torch.randn(4,2),
              'T2.std.sw':torch.tensor([.5,2.]),'T2.teacher':torch.randn(1,2)}
        actual=alignment(self.model,self.row,0,'T2',maps)
        h=manual_states(self.model,self.row)[17][[0,2]].mean(0)
        expected=((h-maps['body.mean.sw'])@maps['T2.projection.sw']/maps['T2.std.sw']-maps['T2.teacher'][0]).square().mean()
        torch.testing.assert_close(actual,expected)
        a=torch.autograd.grad(actual,list(self.model.parameters()))
        b=torch.autograd.grad(expected,list(self.model.parameters()))
        for x,y in zip(a,b):torch.testing.assert_close(x,y)

    def test_mode_restoration_on_exception(self):
        self.model.train();self.model.model.layers[0].eval()
        before=[m.training for m in self.model.modules()]
        with self.assertRaises(RuntimeError):
            with eval_grad(self.model):
                self.assertFalse(any(m.training for m in self.model.modules()))
                raise RuntimeError('test')
        self.assertEqual(before,[m.training for m in self.model.modules()])

    def test_one_auxiliary_exposure_per_optimizer_step(self):
        # Simulate eight microbatches. Aux gradient must occur exactly once,
        # with no additional division by the target accumulation factor.
        assets=Assets([self.row],{'a':self.ref},{'math':2.},[],{},0.)
        expected=torch.autograd.grad(old_objective(self.model,self.row,'full_feature',self.ref,2.),list(self.model.parameters()))
        class Upstream:
            def __init__(self,path):
                self.accelerator=SimpleNamespace(gradient_accumulation_steps=1,backward=lambda x:x.backward())
                self.args=SimpleNamespace(world_size=1,output_dir=path)
                self.state=SimpleNamespace(global_step=0)
            def compute_loss_context_manager(self):return contextlib.nullcontext()
            def training_step(self,model,inputs,num_items_in_batch=None):return torch.tensor(0.)
        with tempfile.TemporaryDirectory() as tmp:
            trainer=make_trainer(Upstream,assets,'T0')(tmp)
            for _ in range(8):trainer.training_step(self.model,{})
            self.assertEqual(len(trainer.aux),1)
            for p,e in zip(self.model.parameters(),expected):torch.testing.assert_close(p.grad,e)
            trainer.state.global_step=1
            trainer.training_step(self.model,{})
            self.assertEqual(len(trainer.aux),2)
            for p,e in zip(self.model.parameters(),expected):torch.testing.assert_close(p.grad,2*e)

    def test_cca_retains_known_paired_signal(self):
        torch.set_num_threads(2)
        x=torch.randn(180,140,dtype=torch.float64);x-=x.mean(0)
        y=x.clone()
        a,b,s=cca(x,y)
        self.assertEqual(tuple(a.shape),(140,64))
        self.assertTrue(torch.isfinite(s).all())
        torch.testing.assert_close(x@a,y@b,rtol=1e-7,atol=1e-7)

    def test_jsonl_unicode_line_separator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rows.jsonl'
            path.write_text(json.dumps({'text':'a\u2028b'},ensure_ascii=False)+'\n'+json.dumps({'text':'c'})+'\n')
            self.assertEqual(len(read_jsonl(path)),2)


if __name__=='__main__':unittest.main()
