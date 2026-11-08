"""Frozen-weight, fixed-prefix representation primitives; not a training method."""
from contextlib import contextmanager
import argparse
import torch
import torch.nn.functional as F


def causal_model(model):
    return model.get_base_model() if hasattr(model, 'get_base_model') else model


@contextmanager
def block_outputs(model, require_graph=False):
    raw = causal_model(model)
    assert not any(p.requires_grad for p in raw.parameters()), 'Analysis requires frozen parameters'
    values = {}
    handles = []
    for index, block in enumerate(raw.model.layers):
        def hook(module, inputs, output, layer=index):
            values[layer] = output[0] if isinstance(output, tuple) else output
        handles.append(block.register_forward_hook(hook))
    if require_graph:
        handles.append(raw.get_input_embeddings().register_forward_hook(lambda module, inputs, output: output.requires_grad_(True)))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def assistant_forward(model, ids, prefix_length, require_graph=False):
    """CE on assistant labels, with logits computed only at their prediction positions."""
    raw = causal_model(model)
    assert ids.ndim == 2 and ids.shape[0] == 1 and 0 < prefix_length < ids.shape[1]
    with block_outputs(raw, require_graph) as states:
        last = raw.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    assert sorted(states) == list(range(len(raw.model.layers)))
    logits = raw.lm_head(last[:, prefix_length - 1:-1, :])
    labels = ids[:, prefix_length:]
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return loss, [states[index] for index in range(len(states))]


def gradient_basis(vectors, rank=8):
    """Uncentered top gradient directions via the smaller sample Gram matrix."""
    assert vectors.ndim == 2 and vectors.shape[0] >= rank and vectors.shape[1] >= rank
    g = vectors.detach().cpu().double()
    eigenvalues, u = torch.linalg.eigh(g @ g.T)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    u = u[:, order]
    assert eigenvalues[rank - 1] > eigenvalues[0] * 1e-10, 'Insufficient nonzero gradient rank'
    q = g.T @ u[:, :rank] / eigenvalues[:rank].sqrt()
    q = torch.linalg.qr(q, mode='reduced').Q
    assert torch.allclose(q.T @ q, torch.eye(rank, dtype=q.dtype), atol=1e-8)
    return q.float(), float(eigenvalues[:rank].sum() / eigenvalues.sum())


def random_basis(dimension, rank, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.linalg.qr(torch.randn(dimension, rank, generator=generator, dtype=torch.float64), mode='reduced').Q.float()


def drift_statistics(base, adapted, sensitive, random):
    """Inputs are matching positions from identical teacher-forced token sequences."""
    assert base.shape == adapted.shape and base.ndim == 2
    assert sensitive.shape == random.shape and sensitive.shape[0] == base.shape[-1]
    h = base.double()
    difference = adapted.double() - h
    denominator = h.square().sum().clamp_min(1e-12)
    total = difference.square().sum()
    functional = (difference @ sensitive.double()).square().sum()
    control = (difference @ random.double()).square().sum()
    return {'total_normalized_drift': float(total / denominator),
            'functional_normalized_drift': float(functional / denominator),
            'random_normalized_drift': float(control / denominator),
            'functional_fraction_of_drift': float(functional / total.clamp_min(1e-12)),
            'random_fraction_of_drift': float(control / total.clamp_min(1e-12)),
            'reference_squared_energy': float(denominator), 'positions': len(h)}


def validate():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=128, attention_dropout=0)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    ids = torch.tensor([[3, 5, 7, 11, 13, 17, 19, 23]])
    loss, states = assistant_forward(model, ids, 4, True)
    gradients = torch.autograd.grad(loss, states)
    labels = ids.clone()
    labels[:, :4] = -100
    with block_outputs(model, True) as oracle_states:
        oracle = model(input_ids=ids, labels=labels, use_cache=False).loss
    oracle_gradients = torch.autograd.grad(oracle, [oracle_states[k] for k in range(3)])
    assert torch.allclose(loss, oracle, atol=1e-6)
    assert all(torch.allclose(a, b, atol=1e-7, rtol=1e-5) for a, b in zip(gradients, oracle_gradients))
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    q, energy = gradient_basis(torch.randn(48, 32), 8)
    reference = torch.randn(4, 32)
    control = random_basis(32, 8, 42)
    zero = drift_statistics(reference, reference, q, control)
    assert all(zero[k] == 0 for k in ['total_normalized_drift', 'functional_normalized_drift', 'random_normalized_drift'])
    perturbation = reference + q[:, 0].repeat(4, 1)
    measured = drift_statistics(reference, perturbation, q, control)
    assert abs(measured['functional_fraction_of_drift'] - 1) < 1e-6 and 0 < energy <= 1
    assert all(len(block._forward_hooks) == 0 for block in model.model.layers)
    print('CPU Qwen3 oracles passed: assistant CE, all block gradients, frozen weights, hook cleanup, basis orthogonality, zero drift and known-direction drift; no GPU use')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true', required=True)
    parser.parse_args()
    validate()
