"""Shape contracts: every tensor that flows through the model must have the
shape the annotations claim. These tests are cheap and catch the most common
class of bug when editing transformer code (a stray transpose / reshape)."""

import torch

from glassbox import GPT, GPTConfig, MultiHeadAttention, generate


def tiny_cfg(**kw):
    base = dict(vocab_size=17, block_size=12, n_layer=2, n_head=3, n_embd=24)
    base.update(kw)
    return GPTConfig(**base)


def test_multihead_attention_shape():
    torch.manual_seed(0)
    B, T, C = 4, 7, 24
    mha = MultiHeadAttention(n_embd=C, n_head=3, block_size=16)
    x = torch.randn(B, T, C)
    y = mha(x)
    assert y.shape == (B, T, C)          # attention is shape-preserving


def test_gpt_forward_logits_and_loss():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = GPT(cfg)
    B, T = 4, cfg.block_size
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss.ndim == 0                      # scalar
    # A freshly initialised model over V classes should have loss ~ ln(V).
    import math
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5


def test_gpt_forward_without_targets_returns_none_loss():
    cfg = tiny_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    logits, loss = model(idx)
    assert logits.shape == (2, 5, cfg.vocab_size)
    assert loss is None


def test_generate_extends_by_exactly_max_new_tokens():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = GPT(cfg)
    idx = torch.zeros((2, 1), dtype=torch.long)     # (B, T0=1)
    gen = torch.Generator().manual_seed(42)
    out = generate(model, idx, max_new_tokens=20, temperature=1.0, top_k=5, generator=gen)
    assert out.shape == (2, 21)                      # 1 + 20
    assert out.dtype == torch.long
    # Every produced id must be a valid vocab index.
    assert int(out.min()) >= 0 and int(out.max()) < cfg.vocab_size


def test_generate_longer_than_block_size_is_ok():
    """Generation must keep working past block_size by cropping the context."""
    torch.manual_seed(0)
    cfg = tiny_cfg(block_size=8)
    model = GPT(cfg)
    idx = torch.zeros((1, 1), dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    out = generate(model, idx, max_new_tokens=20, generator=gen)   # 21 > block_size 8
    assert out.shape == (1, 21)
