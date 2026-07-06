"""Model & training hyper-parameters, collected in one small dataclass.

Keeping every knob in one place means the rest of the code can be read top to
bottom without hunting for magic numbers, and the README can point at a single
struct when it explains what each dimension means.

Naming convention used throughout the codebase (memorise these five letters and
every shape comment becomes readable):

    B  = batch size            (how many sequences we process at once)
    T  = time / block size     (sequence length, number of tokens)
    C  = n_embd                (the model / embedding dimension, a.k.a. d_model)
    nh = n_head                (number of attention heads)
    hs = head size = C // nh   (per-head dimension, a.k.a. d_k)

So `C == nh * hs` always. A shape comment like `(B, T, C) -> (B, nh, T, hs)`
is literally "reshape the channel axis C into nh heads of size hs and move the
head axis next to the batch axis".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- vocabulary ---
    vocab_size: int = 65          # set from the tokenizer at build time

    # --- architecture ---
    block_size: int = 128         # T: maximum context length in tokens
    n_layer: int = 4              # number of stacked transformer blocks
    n_head: int = 4               # nh: number of attention heads
    n_embd: int = 128             # C: model width (must be divisible by n_head)
    dropout: float = 0.0          # 0.0 keeps the tiny demos deterministic & easy to overfit

    # --- misc ---
    bias: bool = True             # use bias terms in Linear / LayerNorm layers

    def __post_init__(self) -> None:
        # A silent mismatch here produces very confusing shape errors deep inside
        # attention, so we fail loudly and early instead.
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}); "
            f"head size = n_embd / n_head must be an integer."
        )

    @property
    def head_size(self) -> int:
        """hs = C // nh — the width of each individual attention head."""
        return self.n_embd // self.n_head
