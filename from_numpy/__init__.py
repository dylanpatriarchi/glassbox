"""Pure-NumPy attention with a hand-derived backward pass.

This module exists to prove one thing: that we understand *backpropagation
through attention*, not just how to call `loss.backward()`. Everything here is
NumPy — no autograd. The forward pass and the analytic gradients are both
written by hand, then checked against numerical (finite-difference) gradients.

See `attention_numpy.py` for the full derivation in the comments.
"""

from .attention_numpy import (
    attention_backward,
    attention_forward,
    numerical_gradient,
    softmax,
)

__all__ = [
    "attention_forward",
    "attention_backward",
    "softmax",
    "numerical_gradient",
]
