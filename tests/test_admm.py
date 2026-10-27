"""Tests of canonical ADMM on the closed-form two-block problem of test_albcd.py:

    min (x0 - a)^2 + (x1 - b)^2  s.t.  phi = x0 - x1 = 0,  solution x0 = x1 = (a + b) / 2, y = a - b.
"""

import numpy as np
import pytest

from albcd import ADMM
from test_albcd import Block, A, B


def blocks():
    return [Block(0, A), Block(1, B)]


@pytest.mark.parametrize("mu", [0.3, 1.0, 10.0])
def test_converges_to_known_solution(mu):
    opt = ADMM(blocks(), np.zeros(2), mu, n_con=1, feas_tol=1e-9, opt_tol=1e-9, verbose=False)
    assert opt.solve() is opt
    assert opt.success
    np.testing.assert_allclose(opt.x, [(A + B) / 2] * 2, atol=1e-8)
    np.testing.assert_allclose(opt.y, [A - B], atol=1e-7)


def test_penalty_is_fixed():
    mu = np.array([2.0])
    opt = ADMM(blocks(), np.zeros(2), mu, feas_tol=1e-9, opt_tol=1e-9, verbose=False).solve()
    assert opt.mu[0] == 2.0 and mu[0] == 2.0  # never updated, and the caller's array is not aliased


def test_stops_on_both_residuals():
    opt = ADMM(blocks(), np.zeros(2), 1.0, n_con=1, feas_tol=1e-6, opt_tol=1e-6, verbose=False).solve()
    assert opt.feas_history[-1] <= 1e-6 and opt.opt_history[-1] <= 1e-6
    assert all(f > 1e-6 or o > 1e-6 for f, o in zip(opt.feas_history[:-1], opt.opt_history[:-1]))


def test_one_sweep_per_iteration_and_histories():
    opt = ADMM(blocks(), np.zeros(2), 1.0, n_con=1, feas_tol=1e-8, opt_tol=1e-8, verbose=False).solve()
    assert len(opt.x_history) - 1 == 2 * len(opt.feas_history) == 2 * len(opt.opt_history)
    np.testing.assert_array_equal(opt.x_history[-1], opt.x)


def test_not_converged_and_errors():
    opt = ADMM(blocks(), np.zeros(2), 1.0, n_con=1, feas_tol=1e-12, opt_tol=1e-12, max_iter=2, verbose=False)
    assert not opt.solve().success
    with pytest.raises(ValueError):
        ADMM(blocks(), np.zeros(2), 1.0)  # a scalar mu needs n_con
