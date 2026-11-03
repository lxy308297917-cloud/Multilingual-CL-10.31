"""Check actual low-rank forward and autograd against a dense intervention."""
import unittest
import sys
from pathlib import Path
if Path(__file__).parent.name == 'research_my_method':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/research_my_method'))
import torch
from pilot_semantic_coefficients import CoefficientLinear


class CoefficientForwardTest(unittest.TestCase):
    def test_qwen_checkpointed_backward_with_frozen_base(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(29)
        config = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=8, tie_word_embeddings=True)
        config._attn_implementation = 'eager'
        model = Qwen3ForCausalLM(config)
        model.requires_grad_(False)
        wrappers = []
        for layer in model.model.layers:
            for parent, names in [(layer.self_attn, ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
                                  (layer.mlp, ['gate_proj', 'up_proj', 'down_proj'])]:
                for name in names:
                    linear = getattr(parent, name)
                    m, n = linear.weight.shape
                    u = torch.linalg.qr(torch.randn(m, 3), mode='reduced').Q
                    v = torch.linalg.qr(torch.randn(n, 3), mode='reduced').Q
                    wrapped = CoefficientLinear(linear, {'u': u, 'v': v, 's': torch.ones(3)})
                    setattr(parent, name, wrapped)
                    wrappers.append(wrapped)
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        scores = model(input_ids=torch.tensor([[1, 5, 3, 8]]), use_cache=False,
                       logits_to_keep=1).logits[0, -1, [10, 11, 12, 13]]
        torch.nn.functional.cross_entropy(scores[None], torch.tensor([2])).backward()
        self.assertEqual(len(wrappers), 14)
        for wrapped in wrappers:
            self.assertIsNotNone(wrapped.coefficient.grad)
            self.assertTrue(torch.isfinite(wrapped.coefficient.grad).all())
            self.assertGreater(float(wrapped.coefficient.grad.norm()), 0)
            self.assertIsNone(wrapped.original.weight.grad)

    def test_forward_gradient_and_effective_norm(self):
        torch.manual_seed(23)
        u = torch.linalg.qr(torch.randn(9, 3), mode='reduced').Q
        v = torch.linalg.qr(torch.randn(7, 3), mode='reduced').Q
        layer = torch.nn.Linear(7, 9)
        layer.requires_grad_(False)
        wrapped = CoefficientLinear(layer, {'u': u.requires_grad_(), 'v': v.requires_grad_(), 's': torch.ones(3)})
        self.assertFalse(wrapped.u.requires_grad)
        self.assertFalse(wrapped.v.requires_grad)
        with torch.no_grad():
            wrapped.coefficient.copy_(torch.tensor([.2, -.1, .3]))
        x = torch.randn(2, 5, 7)
        coef = wrapped.coefficient.detach().clone().requires_grad_()
        delta = (u*coef)@v.T
        dense = torch.nn.functional.linear(x, layer.weight+delta, layer.bias)
        actual = wrapped(x)
        torch.testing.assert_close(actual, dense, rtol=2e-6, atol=3e-7)
        upstream = torch.randn_like(actual)
        (actual*upstream).sum().backward()
        (dense*upstream).sum().backward()
        torch.testing.assert_close(wrapped.coefficient.grad, coef.grad, rtol=2e-6, atol=2e-6)
        self.assertIsNone(layer.weight.grad)
        c = coef.detach().double()
        gram = (u.double().T@u.double())*(v.double().T@v.double())
        self.assertAlmostEqual(float(c@gram@c), float(delta.detach().double().square().sum()), places=7)
        before = layer.weight.detach().clone()
        with torch.no_grad():
            wrapped.coefficient.zero_()
        torch.testing.assert_close(wrapped(x), layer(x), rtol=0, atol=0)
        self.assertTrue(torch.equal(before, layer.weight))


if __name__ == '__main__':
    unittest.main()
