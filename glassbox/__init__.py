"""Glassbox — a decoder-only transformer you can fully understand.

Public surface, re-exported for convenience:

    from glassbox import GPT, GPTConfig, CharTokenizer, generate

Every module is small and single-purpose; read them in this order:
    config.py -> tokenizer.py -> attention.py -> model.py -> sample.py -> train.py
"""

from .attention import MultiHeadAttention, scaled_dot_product_attention
from .config import GPTConfig
from .data import TextData, sorted_copy_batch
from .model import GPT, Block, LayerNorm, MLP
from .sample import generate
from .tokenizer import CharTokenizer

__all__ = [
    "GPTConfig",
    "CharTokenizer",
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "GPT",
    "Block",
    "LayerNorm",
    "MLP",
    "generate",
    "TextData",
    "sorted_copy_batch",
]

__version__ = "0.1.0"
