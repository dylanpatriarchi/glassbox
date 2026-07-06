"""The 'can it learn?' sanity test.

A correctly wired model with a working backward pass can memorise a single tiny
batch, driving the loss to ~zero. If the forward/backward path were broken (a
detached tensor, a wrong reshape, a mask leaking the future), the loss would
plateau. So near-zero loss on one fixed batch is strong evidence the gradients
flow correctly end to end.

This is deliberately fast (a few hundred steps on a tiny model, CPU-only).
"""

import torch

from glassbox import GPT, GPTConfig


def test_overfits_single_batch():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=13, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)

    # One fixed batch, used every step. x is a random token sequence; y is the
    # next-token target (x shifted left by one). Because it never changes, the
    # model can memorise the exact mapping and reach near-zero loss.
    B, T = 4, cfg.block_size
    g = torch.Generator().manual_seed(0)
    seq = torch.randint(0, cfg.vocab_size, (B, T + 1), generator=g)
    x = seq[:, :-1]          # (B, T)
    y = seq[:, 1:]           # (B, T)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()

    losses = []
    for _ in range(400):
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(loss.item())

    # The loss must drop dramatically and end very small.
    assert losses[-1] < 0.02, f"final loss {losses[-1]:.4f} — model did not overfit"
    assert losses[-1] < losses[0] * 0.05, "loss did not decrease by ~20x"
