"""Datasets: a real char-level text corpus and a crisp synthetic task.

We provide two data sources on purpose, because they teach different things:

1. `TextData` — char-level language modelling on a bundled text file (a small
   public-domain Shakespeare excerpt ships in `data/tiny_shakespeare.txt`). This
   is the classic demo: the model learns the statistics of English/Shakespeare
   and generates plausible-looking text. Great for the "before vs after" samples.

2. `sorted_copy_batch` — a synthetic task where the target is the sorted version
   of the input sequence. It has a *known correct answer*, so "did the model
   learn?" becomes unambiguous (we can measure exact-match accuracy, not just a
   fuzzy loss). We use it in the fast unit tests.

Both produce integer token-id tensors, which is all the model consumes.
"""

from __future__ import annotations

import os

import torch

from .tokenizer import CharTokenizer

# Absolute path to the bundled corpus, resolved relative to this file so it works
# no matter what the current working directory is.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.join(_HERE, "..", "data", "tiny_shakespeare.txt")


class TextData:
    """Wraps a text corpus: tokenizes it and serves random (x, y) batches.

    The train/val split is a simple 90/10 cut of the token stream. For each batch
    we pick B random start positions and slice out a window of `block_size`
    tokens as the input x, and the *same window shifted right by one* as the
    target y. That shift is the essence of language modelling: at every position,
    predict the next token.
    """

    def __init__(self, path: str = DEFAULT_CORPUS, block_size: int = 128,
                 train_frac: float = 0.9, device: str = "cpu") -> None:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokenizer = CharTokenizer(text)
        self.vocab_size = self.tokenizer.vocab_size
        self.block_size = block_size
        self.device = device

        # Encode the whole corpus once into a flat 1-D tensor of token ids.
        data = self.tokenizer.encode_to_tensor(text)         # (N,)
        n = int(train_frac * len(data))
        self.train_data = data[:n]                           # (N_train,)
        self.val_data = data[n:]                             # (N_val,)

    def get_batch(self, split: str, batch_size: int,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y), each of shape (batch_size, block_size)."""
        data = self.train_data if split == "train" else self.val_data
        # Highest valid start index so that a full block + its shifted target fit.
        high = len(data) - self.block_size - 1
        # Guard: a corpus (or split) shorter than block_size+2 tokens leaves no
        # room to sample a window, and torch.randint would raise a cryptic error.
        # Fail loudly with an actionable message instead.
        assert high > 0, (
            f"'{split}' split has only {len(data)} tokens but block_size="
            f"{self.block_size}; use a longer corpus or a smaller block_size "
            f"(need > block_size + 1 tokens)."
        )
        ix = torch.randint(high, (batch_size,), generator=generator)  # (B,)

        # Stack the sliced windows into a batch.
        x = torch.stack([data[i : i + self.block_size] for i in ix])          # (B, T)
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])  # (B, T)
        return x.to(self.device), y.to(self.device)


# --- Synthetic task: sorted-copy -------------------------------------------------

# A tiny fixed vocabulary of digit characters "0".."9" for the synthetic task.
SORT_VOCAB = [str(d) for d in range(10)]
SORT_STOI = {c: i for i, c in enumerate(SORT_VOCAB)}


def sorted_copy_batch(
    batch_size: int, seq_len: int, n_symbols: int = 10,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch for the 'sort these digits' task, framed for a CAUSAL LM.

    A subtlety trips up the naive version: you cannot align a random sequence
    `x[t]` with its sorted counterpart `sorted(x)[t]` position-by-position and
    ask a causal model to map one to the other, because `sorted(x)[0]` is the
    global minimum — it depends on the WHOLE input, which position 0 hasn't seen
    yet. A causal model literally cannot solve that.

    The correct framing models the *concatenated* stream `[input | sorted]` and
    trains standard next-token prediction on it. After the model has read all
    `seq_len` input digits, it has enough information to emit the sorted run:

        full = [ d_0, d_1, ..., d_{L-1},   s_0, s_1, ..., s_{L-1} ]   # length 2L
                └────── random ───────┘   └─── sorted(random) ───┘
        x = full[:, :-1]     # (B, 2L-1)  inputs
        y = full[:, 1:]      # (B, 2L-1)  next-token targets

    Only the predictions over the second half (the sorted region) are actually
    determined; a trained model reaching ~100% accuracy THERE is unambiguous
    proof of learning. Because the answer is deterministic, this is an easy,
    fast, self-checking task with no corpus download.
    """
    rand = torch.randint(0, n_symbols, (batch_size, seq_len), generator=generator)  # (B, L)
    srt, _ = torch.sort(rand, dim=1)                                                # (B, L)
    full = torch.cat([rand, srt], dim=1)     # (B, 2L)  the joint [input | sorted] stream
    x = full[:, :-1]                         # (B, 2L-1)
    y = full[:, 1:]                          # (B, 2L-1)
    return x, y
