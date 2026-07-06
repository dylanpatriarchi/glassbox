"""Attention, from scratch — the single most important file in this repo.

We build two things:

1. `scaled_dot_product_attention(...)` — the raw mathematical operation, written
   with plain tensor ops so you can read every step. It is deliberately written
   to match the semantics of `torch.nn.functional.scaled_dot_product_attention`
   so a unit test can assert the two are `allclose` (see tests/).

2. `MultiHeadAttention` — an `nn.Module` that (a) projects the input into
   queries/keys/values, (b) splits the channel dimension into several heads,
   (c) runs scaled dot-product attention per head in parallel, and (d) recombines
   the heads and projects back. The tensor shape is annotated at EVERY step.

--------------------------------------------------------------------------------
WHAT IS ATTENTION, IN ONE PARAGRAPH
--------------------------------------------------------------------------------
Each position in the sequence emits a *query* ("what am I looking for?"), and
every position also exposes a *key* ("what do I contain?") and a *value* ("what
will I hand over if you attend to me?"). We score how well each query matches
each key with a dot product, turn those scores into a probability distribution
with softmax, and then take a weighted average of the values. The result for a
position is a blend of the values of the positions it decided were relevant.

--------------------------------------------------------------------------------
WHY DIVIDE BY sqrt(d_k)?
--------------------------------------------------------------------------------
q and k are vectors of dimension d_k (== head size). If their components are
roughly independent with unit variance, their dot product q·k is a sum of d_k
such products, so it has variance ~ d_k and standard deviation ~ sqrt(d_k). As
d_k grows the raw scores get large in magnitude; feeding large logits into
softmax pushes it into a near one-hot regime where gradients vanish (softmax
saturates). Dividing by sqrt(d_k) rescales the scores back to ~unit variance,
keeping softmax in a well-behaved, high-gradient region. This is exactly the
`1/sqrt(d_k)` factor from "Attention Is All You Need" (Vaswani et al., 2017).

--------------------------------------------------------------------------------
THE CAUSAL MASK
--------------------------------------------------------------------------------
This is a decoder / language model: position t may only attend to positions
<= t, never to the future (otherwise predicting token t+1 would be trivial — the
model could just look at it). We enforce this by setting the pre-softmax score
S[i, j] to -inf for every j > i. softmax(-inf) = 0, so future positions get
exactly zero weight. The mask is a lower-triangular pattern.

Inspired by (but independently written, not copied from) nanoGPT / minGPT.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,          # (..., T_q, hs)  queries
    k: torch.Tensor,          # (..., T_k, hs)  keys
    v: torch.Tensor,          # (..., T_k, hs)  values
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Scaled dot-product attention, written by hand.

    Shapes (the leading `...` is any number of batch-like dims, e.g. (B, nh)):
        q: (..., T_q, hs)
        k: (..., T_k, hs)
        v: (..., T_k, hs)
    returns:
        out: (..., T_q, hs)

    Semantics deliberately mirror `F.scaled_dot_product_attention` so we can
    unit-test the two against each other.
    """
    hs = q.size(-1)  # head size == d_k

    # 1) SCORES: how much each query matches each key.
    #    q @ k^T contracts the hs axis, leaving one score per (query, key) pair.
    #    (..., T_q, hs) @ (..., hs, T_k) -> (..., T_q, T_k)
    scores = q @ k.transpose(-2, -1)

    # 2) SCALE by 1/sqrt(d_k) (see the long comment at the top of the file).
    scores = scores / math.sqrt(hs)                          # (..., T_q, T_k)

    # 3) MASKING. Two independent ways to forbid attending to certain positions:
    #    - is_causal: build a lower-triangular mask on the fly (no future).
    #    - attn_mask: caller-supplied. A bool mask marks *allowed* positions;
    #      a float mask is added to the scores (use -inf to forbid). This matches
    #      torch's convention exactly.
    if is_causal:
        assert attn_mask is None, "pass either is_causal or attn_mask, not both"
        T_q, T_k = scores.size(-2), scores.size(-1)
        # Lower-triangular ones -> True on/below the diagonal (allowed positions).
        causal = torch.ones(T_q, T_k, dtype=torch.bool, device=scores.device).tril()
        # Wherever NOT allowed (the strict upper triangle), set score to -inf.
        scores = scores.masked_fill(~causal, float("-inf"))  # (..., T_q, T_k)
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # True = keep, False = forbid.
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            # Float mask is *added* (e.g. large negative numbers forbid).
            scores = scores + attn_mask

    # 4) SOFTMAX over the *key* axis -> a probability distribution per query.
    #    Each row of `attn` sums to 1: "how do I split my attention budget?"
    attn = F.softmax(scores, dim=-1)                         # (..., T_q, T_k)

    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    # 5) WEIGHTED SUM of values. attn @ v averages the value vectors using the
    #    attention weights.
    #    (..., T_q, T_k) @ (..., T_k, hs) -> (..., T_q, hs)
    out = attn @ v
    return out                                              # (..., T_q, hs)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention.

    Instead of doing attention once with the full width C, we split C into `nh`
    independent heads of size `hs = C // nh` and attend `nh` times in parallel.
    Different heads can specialise (one tracks the previous token, another tracks
    the subject of the sentence, etc.). We then concatenate the heads back to
    width C and apply an output projection to mix them.

    Every intermediate shape is annotated. Read the shape comments like a story.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int,
                 dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head            # hs

        # ONE fused linear layer produces q, k and v together for efficiency:
        # it maps C -> 3C, and we split the output into three C-wide chunks.
        # (Mathematically identical to three separate C->C projections.)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # Output projection, mixes information across heads: C -> C.
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

        # A cached lower-triangular causal mask, stored as a buffer (moves with
        # the module to GPU, is saved in the state_dict, but is NOT a parameter).
        # Shape (1, 1, T, T) so it broadcasts over the (B, nh) leading dims.
        mask = torch.ones(block_size, block_size, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape

        # 1) PROJECT to q, k, v in one matmul, then split the last axis into 3.
        #    (B, T, C) -> (B, T, 3C) --split--> three tensors of (B, T, C)
        qkv = self.c_attn(x)                                  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)               # each (B, T, C)

        # 2) SPLIT each of q, k, v into heads and move the head axis next to B.
        #    (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        #    The transpose(1, 2) is what lets attention run independently per head
        #    (the batch-like leading dims for our SDPA are then (B, nh)).
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)

        # 3) ATTEND, per head, with a causal mask. Slice the cached mask to (T, T)
        #    in case the input is shorter than block_size.
        #    (B, nh, T, hs) -> (B, nh, T, hs)
        attn_mask = self.causal_mask[:, :, :T, :T]            # (1, 1, T, T) bool
        y = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )                                                     # (B, nh, T, hs)

        # 4) RECOMBINE heads: undo the transpose and merge (nh, hs) back into C.
        #    (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        #    `.contiguous()` is required because transpose returns a non-contiguous
        #    view and `.view` needs a contiguous buffer to reinterpret.
        y = y.transpose(1, 2).contiguous().view(B, T, C)      # (B, T, C)

        # 5) OUTPUT PROJECTION mixes the per-head information back together.
        #    (B, T, C) -> (B, T, C)
        y = self.resid_dropout(self.c_proj(y))                # (B, T, C)
        return y
