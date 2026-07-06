"""A char-level tokenizer — the simplest tokenizer that can possibly work.

A tokenizer is just a bijection between text and integers. Real LLMs use
sub-word schemes (BPE, SentencePiece) with tens of thousands of tokens, but for
a glass-box model a *character* vocabulary is perfect: it is trivial to
understand (one integer per character), it has a tiny vocab (so the model stays
small), and there is nothing hidden — you can print the whole `stoi` table.

    encode:  "hello" -> [46, 43, 50, 50, 53]
    decode:  [46, 43, 50, 50, 53] -> "hello"

The only real design decision is *how we build the vocabulary*: we take the set
of unique characters that appear in the training text, sort them (so the mapping
is deterministic across runs / machines), and assign each an index.
"""

from __future__ import annotations

import torch


class CharTokenizer:
    """Maps individual characters <-> integer ids built from a corpus."""

    def __init__(self, text: str) -> None:
        # `sorted(set(text))` is the whole vocabulary. Sorting makes the id
        # assignment deterministic: the same corpus always yields the same table,
        # which matters for reproducibility and for reloading a trained model.
        chars = sorted(set(text))
        self.vocab_size: int = len(chars)

        # Two lookup tables: string-to-int and int-to-string.
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        """Text -> list of token ids. Raises KeyError on unseen characters,
        which is the honest behaviour: a char-level model literally cannot
        represent a symbol it never saw during vocab construction."""
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int]) -> str:
        """List of token ids -> text."""
        return "".join(self.itos[int(i)] for i in ids)

    def encode_to_tensor(self, s: str) -> torch.Tensor:
        """Convenience: text -> 1-D LongTensor of shape (len(s),).

        Token ids must be int64 (`torch.long`) because they are used to index
        into the embedding table, and PyTorch requires long indices there.
        """
        return torch.tensor(self.encode(s), dtype=torch.long)  # (len(s),)
