"""Single-head scaled dot-product attention: forward + MANUAL backward, in NumPy.

The point of this file is the backward pass. If you can derive and verify the
gradients of attention by hand, you understand what autograd is doing for you.

--------------------------------------------------------------------------------
FORWARD
--------------------------------------------------------------------------------
Inputs (single head, no batch, for clarity):
    Q : (T, d)   queries
    K : (T, d)   keys
    V : (T, d)   values

    S = Q @ K^T / sqrt(d)          # (T, T)   scaled scores
    P = softmax(S, axis=-1)        # (T, T)   attention weights, rows sum to 1
    O = P @ V                      # (T, d)   output

--------------------------------------------------------------------------------
BACKWARD
--------------------------------------------------------------------------------
Given the upstream gradient dO = dL/dO (shape (T, d)), we want dQ, dK, dV.

Work backwards through each op:

(1) O = P @ V
        dV = P^T @ dO                              # (T, d)
        dP = dO @ V^T                              # (T, T)

(2) P = softmax(S)  (applied row-wise)
    For a single softmax row p = softmax(s), the Jacobian is
        dp_i/ds_j = p_i (δ_ij - p_j).
    So for an upstream row-gradient g = dL/dp, the down-stream row-gradient is
        dL/ds = p ⊙ (g - (g · p))
    where (g · p) is a scalar (the row's dot product). Vectorised over all rows:
        rowsum = sum(dP ⊙ P, axis=1, keepdims=True)   # (T, 1)
        dS = P ⊙ (dP - rowsum)                          # (T, T)

(3) S = (Q @ K^T) / sqrt(d)
        the 1/sqrt(d) is a constant scale, so dS carries a 1/sqrt(d) factor:
        dScaled = dS / sqrt(d)
        dQ = dScaled @ K                              # (T, d)
        dK = dScaled^T @ Q                            # (T, d)

That's the whole thing. `attention_backward` implements exactly these lines, and
`numerical_gradient` checks them with central finite differences.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtract the max before exponentiating)."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention_forward(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray,
    causal: bool = False,
) -> tuple[np.ndarray, dict]:
    """Forward pass. Returns (O, cache) where cache holds tensors reused in backward.

    Q, K, V : (T, d)
    O       : (T, d)
    """
    T, d = Q.shape
    S = (Q @ K.T) / np.sqrt(d)                     # (T, T)

    if causal:
        # Forbid attending to the future: set strict-upper-triangle to -inf.
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        S = np.where(mask, -np.inf, S)

    P = softmax(S, axis=-1)                        # (T, T)
    O = P @ V                                      # (T, d)

    cache = {"Q": Q, "K": K, "V": V, "P": P, "d": d}
    return O, cache


def attention_backward(
    dO: np.ndarray, cache: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Manual backward pass. Returns (dQ, dK, dV), each (T, d).

    dO : (T, d) upstream gradient of the loss w.r.t. the output O.
    """
    Q, K, V, P, d = cache["Q"], cache["K"], cache["V"], cache["P"], cache["d"]

    # (1) O = P @ V
    dV = P.T @ dO                                  # (T, d)
    dP = dO @ V.T                                  # (T, T)

    # (2) softmax backward, row-wise:  dS = P ⊙ (dP - rowsum(dP ⊙ P))
    rowsum = np.sum(dP * P, axis=1, keepdims=True)  # (T, 1)
    dS = P * (dP - rowsum)                          # (T, T)

    # (3) S = (Q @ K^T) / sqrt(d)
    dScaled = dS / np.sqrt(d)                       # (T, T)
    dQ = dScaled @ K                               # (T, d)
    dK = dScaled.T @ Q                             # (T, d)

    return dQ, dK, dV


def numerical_gradient(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central finite-difference gradient of scalar function f at x.

    grad[i] ≈ (f(x + eps e_i) - f(x - eps e_i)) / (2 eps)

    Used only for testing the analytic gradients above; O(n) forward passes, so
    it is slow and meant for tiny tensors.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        x[idx] = orig  # restore
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad
