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
    """A batch for the 'sort these digits' task.

    Input  x: a random sequence of `seq_len` digit ids, e.g. [3, 1, 2, 1].
    Target y: at each position, the next token of the SORTED sequence, so that a
              standard next-token language model learns to output the sorted
              version. Concretely we model the joint sequence [input | sorted] and
              y is that stream shifted by one; here we expose the simpler
              "predict the sorted sequence" framing used by the overfit test:

                  x = random digits                      (B, L)
                  y = the same digits, sorted ascending  (B, L)

    Because the answer is deterministic, a trained model reaching ~100% token
    accuracy is unambiguous proof of learning.
    """
    x = torch.randint(0, n_symbols, (batch_size, seq_len), generator=generator)  # (B, L)
    y, _ = torch.sort(x, dim=1)                                                   # (B, L)
    return x, y
