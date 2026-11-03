"""Frozen diagnostic scoring; no training or test-fitted parameters."""
import numpy as np
import torch
import torch.nn.functional as F


def permutation(rotation):
    return [(j + rotation) % 4 for j in range(4)]


def remap(values, order):
    out = np.empty(4, dtype=np.float64)
    out[np.asarray(order)] = values
    return out


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    z = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def pride_prior(calibration_scores):
    # PriDe eq7 then mean across independent calibration semantic IDs.
    return softmax(np.asarray(calibration_scores).mean(axis=1)).mean(axis=0)


def corrected(scores, prior):
    return np.asarray(scores) - np.log(np.maximum(prior, 1e-12))


def score_pairs(model, pairs, device, pad_id, batch_size=4):
    """Exact teacher forcing. Compute vocabulary logits only on answer positions.

    Right padding does not change valid position indices; all causal targets are
    shifted by one. Float32 vocabulary projection/logsoftmax is chunked to avoid
    allocating B*prompt_length*vocabulary logits. Backbone follows model dtype.
    """
    sums, lengths = [], []
    for start in range(0, len(pairs), batch_size):
        part = pairs[start:start + batch_size]
        seqs = [(c + a)[:-1] for c, a in part]
        assert all(c and a for c, a in part)
        width = max(map(len, seqs))
        ids = torch.full((len(part), width), pad_id, device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(seqs):
            ids[i, :len(seq)] = torch.tensor(seq, device=device)
            mask[i, :len(seq)] = 1
        hidden = model.model(input_ids=ids, attention_mask=mask,
                             use_cache=False).last_hidden_state
        for i, (context, answer) in enumerate(part):
            positions = hidden[i, len(context) - 1:len(context) + len(answer) - 1]
            targets = torch.tensor(answer, device=device)
            total = 0.
            for k in range(0, len(answer), 32):
                logits = F.linear(positions[k:k + 32].float(),
                                  model.lm_head.weight.float(),
                                  model.lm_head.bias)
                total += float(logits.log_softmax(-1).gather(
                    1, targets[k:k + 32, None]).sum())
            sums.append(total)
            lengths.append(len(answer))
        del hidden
    return sums, lengths


def score_labels(model, pairs_by_rotation, device, pad_id):
    """One prefill per rotation when A/B/C/D share their causal context."""
    contexts, prefix = [], []
    for pairs in pairs_by_rotation:
        assert all(c == pairs[0][0] for c, _ in pairs)
        common = pairs[0][1][:-1]
        assert all(a[:-1] == common for _, a in pairs)
        contexts.append(pairs[0][0] + common)
        prefix.append(common)
    width = max(map(len, contexts))
    ids = torch.full((len(contexts), width), pad_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, c in enumerate(contexts):
        ids[i, :len(c)] = torch.tensor(c, device=device)
        mask[i, :len(c)] = 1
    h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    result = []
    for i, pairs in enumerate(pairs_by_rotation):
        c = pairs[0][0]
        # Common-prefix raw likelihood retained, not discarded.
        pos = h[i, len(c) - 1:len(contexts[i])]
        lp = F.linear(pos.float(), model.lm_head.weight.float(), model.lm_head.bias).log_softmax(-1)
        shared = sum(float(lp[j, t]) for j, t in enumerate(prefix[i]))
        result.append([shared + float(lp[-1, a[-1]]) for _, a in pairs])
    return result
