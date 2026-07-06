"""Our hand-written scaled dot-product attention must match PyTorch's reference.

This is the single most convincing correctness check in the repo: if our
from-scratch `scaled_dot_product_attention` is numerically `allclose` to
`torch.nn.functional.scaled_dot_product_attention` on the same random inputs,
for both the non-causal and causal cases, then the core math is right.
"""

import torch
import torch.nn.functional as F

from glassbox.attention import scaled_dot_product_attention


def _random_qkv(seed: int = 0):
    torch.manual_seed(seed)
    B, nh, T, hs = 2, 3, 5, 8          # (batch, heads, time, head-size)
    q = torch.randn(B, nh, T, hs, dtype=torch.float64)
    k = torch.randn(B, nh, T, hs, dtype=torch.float64)
    v = torch.randn(B, nh, T, hs, dtype=torch.float64)
    return q, k, v


def test_matches_torch_non_causal():
    q, k, v = _random_qkv()
    ours = scaled_dot_product_attention(q, k, v)                     # no mask
    ref = F.scaled_dot_product_attention(q, k, v)                   # torch reference
    assert torch.allclose(ours, ref, atol=1e-10, rtol=1e-8)


def test_matches_torch_causal():
    q, k, v = _random_qkv(seed=1)
    ours = scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-10, rtol=1e-8)


def test_matches_torch_bool_mask():
    """A caller-supplied boolean mask should behave like torch's attn_mask."""
    q, k, v = _random_qkv(seed=2)
    T = q.size(-2)
    # An arbitrary (but valid: at least one True per row) allowed-positions mask.
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    ours = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(ours, ref, atol=1e-10, rtol=1e-8)


def test_causal_actually_hides_the_future():
    """Position 0 must have zero attention to positions > 0. We verify this by
    changing a future value and checking the first output row is unchanged."""
    q, k, v = _random_qkv(seed=3)
    out_a = scaled_dot_product_attention(q, k, v, is_causal=True)
    v2 = v.clone()
    v2[..., 1:, :] += 100.0          # perturb every value EXCEPT position 0
    out_b = scaled_dot_product_attention(q, k, v2, is_causal=True)
    # The first time-step's output depends only on position 0, so it can't move.
    assert torch.allclose(out_a[..., 0, :], out_b[..., 0, :], atol=1e-10)
    # But later positions DO see the perturbed future-of-earlier values, so they move.
    assert not torch.allclose(out_a[..., -1, :], out_b[..., -1, :])
