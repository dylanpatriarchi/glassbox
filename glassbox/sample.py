"""Autoregressive generation: turn a trained model into a text generator.

Generation is a loop: feed the current context, look at the logits for the LAST
position, convert them to a probability distribution over the vocabulary, sample
one token, append it, and repeat. Two standard knobs shape the distribution:

TEMPERATURE (T_temp)
    We divide the logits by a temperature before softmax.
      * T_temp -> 0   : distribution sharpens toward its argmax (greedy, "safe",
                        repetitive).
      * T_temp = 1     : the model's raw distribution.
      * T_temp > 1    : flatter distribution (more random / "creative").

TOP-K
    Keep only the k most-likely tokens and renormalise, zeroing the long tail.
    This prevents the model from occasionally sampling an absurd low-probability
    character, which is especially helpful early in training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import GPT


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,          # (B, T0) starting context (token ids), may be empty-ish
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Autoregressively extend `idx` by `max_new_tokens` tokens.

    Returns a tensor of shape (B, T0 + max_new_tokens).

    `generator` lets callers pass a seeded RNG for fully reproducible sampling.
    """
    model.eval()
    block_size = model.cfg.block_size

    for _ in range(max_new_tokens):
        # 1) CROP the context to the last `block_size` tokens — the model's
        #    learned positional embeddings are only defined up to block_size.
        idx_cond = idx[:, -block_size:]                       # (B, <=block_size)

        # 2) FORWARD pass; we only need the logits at the final time step, since
        #    that is the distribution over the *next* token.
        logits, _ = model(idx_cond)                          # (B, T, vocab)
        logits = logits[:, -1, :]                            # (B, vocab)

        # 3) TEMPERATURE scaling. Guard against divide-by-zero: temperature 0 is
        #    interpreted as greedy (argmax).
        if temperature == 0.0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
            idx = torch.cat((idx, next_id), dim=1)
            continue

        logits = logits / temperature                        # (B, vocab)

        # 4) TOP-K filtering. Set everything outside the top-k to -inf so softmax
        #    gives it exactly zero probability.
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            # kth largest logit per row; anything below it is discarded.
            kth_vals = torch.topk(logits, k, dim=-1).values[:, -1, None]  # (B, 1)
            logits = logits.masked_fill(logits < kth_vals, float("-inf"))

        # 5) SOFTMAX -> probabilities, then SAMPLE one token per sequence.
        probs = F.softmax(logits, dim=-1)                    # (B, vocab)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)

        # 6) APPEND and continue.
        idx = torch.cat((idx, next_id), dim=1)               # (B, T+1)

    return idx
