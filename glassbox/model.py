"""The full decoder-only transformer, assembled from parts.

Layout of one forward pass (shapes on the right):

    idx (token ids)                         (B, T)
      -> token embedding    + position embedding
      -> x                                   (B, T, C)
      -> Block x n_layer                     (B, T, C)   (each block keeps the shape)
      -> final LayerNorm                     (B, T, C)
      -> lm_head (Linear C -> vocab)         (B, T, vocab_size)   == logits

Two design choices are explained inline where they occur:
  * learned vs sinusoidal positional embeddings (we use learned) — see `GPT.__init__`.
  * pre-norm vs post-norm transformer blocks (we use pre-norm) — see `Block`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .config import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm, by hand (so nothing is hidden).

    For each token vector x of width C, we normalise across the C features to
    zero mean and unit variance, then apply a learnable per-feature gain (weight)
    and shift (bias):

        y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

    Note this normalises across *features*, per token — independent of batch and
    sequence length. That is what makes it stable for language models where the
    sequence length varies. `bias` is optional (GPT-2 style allows dropping it).
    """

    def __init__(self, n_embd: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) — reduce over the last axis (the C features).
        mean = x.mean(dim=-1, keepdim=True)                  # (B, T, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)    # (B, T, 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)      # (B, T, C)
        out = x_hat * self.weight                            # (B, T, C) broadcast over C
        if self.bias is not None:
            out = out + self.bias
        return out


class MLP(nn.Module):
    """The per-token feed-forward network (a.k.a. the "position-wise" MLP).

    Attention moves information *between* positions; the MLP then processes each
    position independently, giving the model capacity to transform the mixed
    representation. The standard recipe expands the width by 4x, applies a GELU
    non-linearity, and projects back:

        C -> 4C -> (GELU) -> C
    """

    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)   # C -> 4C
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)  # 4C -> C
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.c_fc(x)              # (B, T, 4C)
        x = F.gelu(x)                 # (B, T, 4C) — smooth non-linearity
        x = self.c_proj(x)            # (B, T, C)
        x = self.dropout(x)           # (B, T, C)
        return x


class Block(nn.Module):
    """One transformer block: attention + MLP, each wrapped in a residual.

    PRE-NORM vs POST-NORM
    ---------------------
    The original 2017 Transformer put LayerNorm *after* the residual add
    ("post-norm"): x = LN(x + Sublayer(x)). Modern GPTs (and this repo) use
    "pre-norm": x = x + Sublayer(LN(x)). Pre-norm keeps a clean, un-normalised
    residual "highway" running straight through the network, which makes deep
    stacks far easier to train (gradients flow through the identity path without
    being repeatedly squashed by normalisation). The trade-off is that the final
    activations can grow, so we apply one more LayerNorm at the very end of the
    model (see `GPT`). We use pre-norm.

        x = x + Attn(LN1(x))
        x = x + MLP (LN2(x))
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = MultiHeadAttention(
            n_embd=cfg.n_embd, n_head=cfg.n_head, block_size=cfg.block_size,
            dropout=cfg.dropout, bias=cfg.bias,
        )
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg.n_embd, dropout=cfg.dropout, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) in and out — a block is shape-preserving.
        x = x + self.attn(self.ln_1(x))   # residual around attention
        x = x + self.mlp(self.ln_2(x))    # residual around the MLP
        return x


class GPT(nn.Module):
    """A small decoder-only transformer language model."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # --- Embeddings ---
        # Token embedding: a lookup table mapping each of `vocab_size` token ids
        # to a C-dim vector. Shape of the table: (vocab_size, C).
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # Positional embedding: attention is permutation-invariant on its own —
        # it has no notion of order — so we must inject "where is this token".
        #
        # LEARNED vs SINUSOIDAL:
        #   * Sinusoidal (the 2017 paper): fixed sin/cos of varying frequency.
        #     No parameters; can in principle extrapolate to longer sequences.
        #   * Learned (GPT-1/2): a trainable table of shape (block_size, C), one
        #     vector per absolute position. Simpler, and empirically as good or
        #     better within the trained context length.
        # We use LEARNED positional embeddings because they are the simplest thing
        # that clearly works and match the GPT lineage this model imitates. The
        # only cost is that positions >= block_size are undefined (we assert on it).
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.drop = nn.Dropout(cfg.dropout)

        # --- The stack of transformer blocks ---
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

        # --- Final norm + language-model head ---
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        # Project each token's C-dim state to a score (logit) for every vocab item.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # WEIGHT TYING: the input embedding and the output projection are forced to
        # share the same weight matrix. Intuitively both relate "token id" and
        # "C-dim vector", just in opposite directions, so tying them saves
        # parameters and tends to improve results (Press & Wolf, 2017).
        self.tok_emb.weight = self.lm_head.weight

        # Initialise weights (GPT-2 style: small normal init).
        self.apply(self._init_weights)

        # A well-known GPT-2 trick: scale down the init of the residual output
        # projections by 1/sqrt(2 * n_layer) so the residual stream variance does
        # not blow up as more layers are added.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Total number of trainable parameters (the tied embedding counts once)."""
        n = sum(p.numel() for p in self.parameters())
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx:     (B, T) int64 token ids
        targets: (B, T) int64 token ids to predict (idx shifted left by one), or None.

        Returns (logits, loss):
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy if targets given, else None.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}; "
            f"learned positional embeddings are only defined up to block_size."
        )

        # Positions 0, 1, ..., T-1 — one per time step.
        pos = torch.arange(T, device=idx.device)             # (T,)

        # Look up token and position vectors and add them. Broadcasting adds the
        # (T, C) position table to every sequence in the (B, T, C) token batch.
        tok = self.tok_emb(idx)                              # (B, T, C)
        pos_e = self.pos_emb(pos)                            # (T, C)
        x = self.drop(tok + pos_e)                           # (B, T, C)

        # Run the transformer blocks; each is shape-preserving.
        for block in self.blocks:
            x = block(x)                                     # (B, T, C)

        x = self.ln_f(x)                                     # (B, T, C)
        logits = self.lm_head(x)                             # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy expects (N, vocab) logits and (N,) targets, so we flatten
            # the batch and time axes together. This is the next-token prediction
            # objective: at every position, predict the following character.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),            # (B*T, vocab_size)
                targets.reshape(-1),                         # (B*T,)
            )
        return logits, loss
