"""Gradient check for the hand-derived NumPy attention backward pass.

We build a scalar loss L = sum(O * G) for a fixed random matrix G, which makes
the upstream gradient dO = dL/dO exactly G. We then compare our analytic dQ, dK,
dV against central finite-difference numerical gradients. Agreement to ~1e-6
means the hand-derived backward math (especially the softmax Jacobian term) is
correct.
"""

import numpy as np

from from_numpy import (
    attention_backward,
    attention_forward,
    numerical_gradient,
)


def _setup(seed: int = 0, causal: bool = False):
    rng = np.random.default_rng(seed)
    T, d = 4, 3
    Q = rng.standard_normal((T, d))
    K = rng.standard_normal((T, d))
    V = rng.standard_normal((T, d))
    G = rng.standard_normal((T, d))     # fixed upstream weights -> dO = G

    def loss_of_output(O):
        return float(np.sum(O * G))

    return Q, K, V, G, causal, loss_of_output


def _check(causal: bool):
    Q, K, V, G, causal, loss_of_output = _setup(causal=causal)

    # Analytic gradients.
    O, cache = attention_forward(Q, K, V, causal=causal)
    dQ, dK, dV = attention_backward(G, cache)   # dO == G

    # Numerical gradients: perturb each input and re-run the forward pass.
    def make_f(which):
        def f(mat):
            if which == "Q":
                Oo, _ = attention_forward(mat, K, V, causal=causal)
            elif which == "K":
                Oo, _ = attention_forward(Q, mat, V, causal=causal)
            else:
                Oo, _ = attention_forward(Q, K, mat, causal=causal)
            return loss_of_output(Oo)
        return f

    dQ_num = numerical_gradient(make_f("Q"), Q.copy())
    dK_num = numerical_gradient(make_f("K"), K.copy())
    dV_num = numerical_gradient(make_f("V"), V.copy())

    assert np.allclose(dQ, dQ_num, atol=1e-6), np.abs(dQ - dQ_num).max()
    assert np.allclose(dK, dK_num, atol=1e-6), np.abs(dK - dK_num).max()
    assert np.allclose(dV, dV_num, atol=1e-6), np.abs(dV - dV_num).max()


def test_backward_matches_numerical_non_causal():
    _check(causal=False)


def test_backward_matches_numerical_causal():
    _check(causal=True)


def test_softmax_rows_sum_to_one():
    from from_numpy import softmax
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 7))
    p = softmax(x, axis=-1)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert (p >= 0).all()
